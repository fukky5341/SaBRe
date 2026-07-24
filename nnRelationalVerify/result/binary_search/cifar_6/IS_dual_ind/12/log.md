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
execution time: IAR + LP analysis = 3.63 + 186.99 = 190.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.4199584, upper bound: 0.4199560


# Binary Search by BASE starts (time budget: 17809.38 seconds, max iter: 100)

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
Binary search time: 1287.64 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Individual Split (IS_dual_ind) starts
Time budget: 16521.74 seconds

## Binary search (step 0) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3357238, upper bound: 0.3379978
time: 247.27 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3396533, upper bound: 0.3396492
time: 811.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1058.53 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1058.53
Output dim: 1, lower bound: -0.3357238, upper bound: 0.3379978
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1058.53
Output dim: 1, lower bound: -0.3396533, upper bound: 0.3396492

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.7581359, 0.3389780, -0.7609042, 0.3420930, -0.4362206, 0.4359148
1: -0.8015226, 0.9825063, -0.8088776, 0.9878285, -1.2603443, 1.2630160
2: -2.7040544, -0.1674972, -2.7020259, -0.1673183, -1.6689529, 1.6669116
3: -7.1978703, -3.3868303, -7.1963382, -3.3863380, -1.5851051, 1.5828753
4: -6.0258999, -2.7192080, -6.0293493, -2.7159114, -1.7157809, 1.7162180
5: -7.5071640, -3.6074045, -7.5031528, -3.6071973, -1.7506578, 1.7479985
6: -10.5196686, -5.7972498, -10.5209427, -5.7922955, -1.6184820, 1.6146799
7: -6.7295203, -1.9411889, -6.7253995, -1.9409115, -3.2766948, 3.2729797
8: -1.2020236, 0.5428982, -1.2036021, 0.5456157, -1.0787117, 1.0775496
9: -1.4420044, 0.4691036, -1.4446292, 0.4691972, -1.6212775, 1.6229117

Time for backsubstitution: 2.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 293
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3487
type: B, layer: 1, pos: 2967
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 2626
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
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 68
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
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 281
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2991
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
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2274
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
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 3260
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
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 3325
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2424
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
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3473
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 3087
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
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2723
type: B, layer: 1, pos: 2722
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 54
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
type: B, layer: 1, pos: 2737
type: B, layer: 1, pos: 3533
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 3223
type: B, layer: 1, pos: 3237
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 3582
type: B, layer: 1, pos: 2076
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 293

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 422

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3357054, upper bound: 0.3370688
time: 96.63 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3357046, upper bound: 0.3379782
time: 212.52 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.7613775, 0.3492614, -0.7613829, 0.3492677, -0.4470952, 0.4364184
1: -0.8265103, 0.9890466, -0.8265154, 0.9890478, -1.2712330, 1.2854685
2: -2.7041450, -0.1663675, -2.7041521, -0.1662247, -1.6742136, 1.6732130
3: -7.1971817, -3.3863952, -7.1971817, -3.3861451, -1.5862737, 1.5869350
4: -6.0309362, -2.7083230, -6.0309448, -2.7083025, -1.7282369, 1.7221004
5: -7.5052214, -3.6084094, -7.5052328, -3.6078382, -1.7538787, 1.7529505
6: -10.5238705, -5.7908487, -10.5238762, -5.7908421, -1.6198031, 1.6236010
7: -6.7265453, -1.9412841, -6.7265663, -1.9410534, -3.2750933, 3.2756603
8: -1.2038989, 0.5512741, -1.2038991, 0.5512744, -1.0862586, 1.0801131
9: -1.4500248, 0.4696424, -1.4500290, 0.4696892, -1.6310222, 1.6301550

Time for backsubstitution: 3.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 293
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3487
type: B, layer: 1, pos: 2967
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 294
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 293

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3395859, upper bound: 0.3368592
time: 253.77 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3396499, upper bound: 0.3396470
time: 80.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 337.75 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 337.75
Output dim: 1, lower bound: -0.3357054, upper bound: 0.3370688
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 337.75
Output dim: 1, lower bound: -0.3357046, upper bound: 0.3379782
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 337.75
Output dim: 1, lower bound: -0.3395859, upper bound: 0.3368592
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 337.75
Output dim: 1, lower bound: -0.3396499, upper bound: 0.3396470

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.7578385, 0.3368762, -0.7587085, 0.3392162, -0.4330733, 0.4317110
1: -0.7946439, 0.9824490, -0.7992724, 0.9848243, -1.2508349, 1.2538276
2: -2.7018452, -0.1791115, -2.6959128, -0.1810619, -1.6525064, 1.6484891
3: -7.1974778, -3.4034705, -7.1898575, -3.4072247, -1.5632048, 1.5593216
4: -6.0243831, -2.7201056, -6.0284991, -2.7169604, -1.7110996, 1.7110298
5: -7.5068426, -3.6213193, -7.4990578, -3.6236527, -1.7325317, 1.7288578
6: -10.5196457, -5.8220692, -10.5120640, -5.8241091, -1.5851051, 1.5796890
7: -6.7255735, -1.9415714, -6.7212601, -1.9437014, -3.2701209, 3.2685761
8: -1.1994528, 0.5422021, -1.2004400, 0.5455103, -1.0755992, 1.0734426
9: -1.4365277, 0.4686172, -1.4369756, 0.4642464, -1.6086642, 1.6148219

Time for backsubstitution: 2.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 293
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
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2642
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
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 296
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
type: A, layer: 1, pos: 422
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
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 68
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
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2991
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
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 3260
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
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 3325
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3473
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 159
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
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2737
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3533
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 3223
type: A, layer: 1, pos: 3237
type: A, layer: 1, pos: 3582
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 2076
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
type: A, layer: 1, pos: 293

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3299349, upper bound: 0.3341386
time: 79.16 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2980

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3353860, upper bound: 0.3366299
time: 186.85 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3352672, upper bound: 0.3366320
time: 28.14 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.7581305, 0.3389757, -0.7608973, 0.3420900, -0.4329418, 0.4359080
1: -0.8014960, 0.9825013, -0.8088409, 0.9878215, -1.2603073, 1.2551742
2: -2.7040453, -0.1675050, -2.7020135, -0.1673290, -1.6627715, 1.6668918
3: -7.1978664, -3.3868613, -7.1963334, -3.3863807, -1.5606426, 1.5828104
4: -6.0258951, -2.7197599, -6.0293441, -2.7166877, -1.7151164, 1.7187299
5: -7.5071573, -3.6074195, -7.5031428, -3.6072173, -1.7295644, 1.7478914
6: -10.5196695, -5.7973156, -10.5209427, -5.7923884, -1.5819237, 1.6145840
7: -6.7294779, -1.9411906, -6.7253437, -1.9409142, -3.2766364, 3.2685444
8: -1.2020144, 0.5428899, -1.2035885, 0.5456042, -1.0786897, 1.0764554
9: -1.4419802, 0.4690995, -1.4445949, 0.4691910, -1.6235681, 1.6222937

Time for backsubstitution: 2.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 293
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
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2642
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
type: A, layer: 1, pos: 422
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
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 68
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
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2991
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
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2743
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 3325
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3473
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 159
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
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2737
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3533
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 3223
type: A, layer: 1, pos: 3237
type: A, layer: 1, pos: 3582
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 2076
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
type: A, layer: 1, pos: 293

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3299347, upper bound: 0.3350470
time: 1094.26 seconds

## Relational analysis of IS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2980

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3353839, upper bound: 0.3375398
time: 511.05 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3352700, upper bound: 0.3375445
time: 282.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 1896.92 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1896.92
Output dim: 1, lower bound: -0.3353860, upper bound: 0.3366299
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1896.92
Output dim: 1, lower bound: -0.3352672, upper bound: 0.3366320
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1896.92
Output dim: 1, lower bound: -0.3353839, upper bound: 0.3375398
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1896.92
Output dim: 1, lower bound: -0.3352700, upper bound: 0.3375445
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1896.92
Output dim: 1, lower bound: -0.3395859, upper bound: 0.3368592
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1896.92
Output dim: 1, lower bound: -0.3396499, upper bound: 0.3396470
Binary search (step 0): status=Status.UNKNOWN, k_low=2, k_high=8, k_mid=5, eps_mid=0.0195312, abs_max=1.2854982614517212
rel_dist={1: [-0.3396559042077183, 0.33965503665474267]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 294

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2823393, upper bound: 0.2837032
time: 130.49 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853745, upper bound: 0.2853644
time: 499.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 629.67 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 629.67
Output dim: 1, lower bound: -0.2823393, upper bound: 0.2837032
IS_A2, status: Status.UNKNOWN, split count: 1, time: 629.67
Output dim: 1, lower bound: -0.2853745, upper bound: 0.2853644

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.7581359, 0.3389780, -0.7608346, 0.3410565, -0.4152128, 0.4158705
1: -0.8015226, 0.9825063, -0.8063337, 0.9876510, -1.2375844, 1.2378138
2: -2.7040544, -0.1674972, -2.7017155, -0.1673727, -1.6384732, 1.6365280
3: -7.1978703, -3.3868303, -7.1962147, -3.3864281, -1.5180496, 1.5159173
4: -6.0258999, -2.7192080, -6.0291147, -2.7170060, -1.6619656, 1.6632297
5: -7.5071640, -3.6074045, -7.5028486, -3.6072536, -1.6827309, 1.6802362
6: -10.5196686, -5.7972498, -10.5205164, -5.7925100, -1.5279139, 1.5238218
7: -6.7295203, -1.9411889, -6.7252350, -1.9409477, -3.2254369, 3.2216287
8: -1.2020236, 0.5428982, -1.2035592, 0.5447999, -1.0547348, 1.0543374
9: -1.4420044, 0.4691036, -1.4438523, 0.4691488, -1.6087610, 1.6098976

Time for backsubstitution: 3.26 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2642
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
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 296
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 561
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
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 68
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
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 281
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2991
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
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2274
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
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 3260
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
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 3325
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3473
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 3087
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
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2723
type: B, layer: 1, pos: 2722
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 2737
type: B, layer: 1, pos: 3533
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 3223
type: B, layer: 1, pos: 3237
type: B, layer: 1, pos: 3582
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 2076
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
type: B, layer: 1, pos: 293

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 422

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2823243, upper bound: 0.2831441
time: 221.75 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2823318, upper bound: 0.2836936
time: 90.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.7613775, 0.3492614, -0.7613810, 0.3492653, -0.4271458, 0.4161828
1: -0.8265103, 0.9890466, -0.8265135, 0.9890474, -1.2481589, 1.2627776
2: -2.7041450, -0.1663675, -2.7041492, -0.1662793, -1.6441967, 1.6432199
3: -7.1971817, -3.3863952, -7.1971822, -3.3862417, -1.5194075, 1.5201576
4: -6.0309362, -2.7083230, -6.0309415, -2.7083106, -1.6753983, 1.6691180
5: -7.5052214, -3.6084094, -7.5052290, -3.6080689, -1.6862915, 1.6855161
6: -10.5238705, -5.7908487, -10.5238743, -5.7908440, -1.5292745, 1.5331656
7: -6.7265453, -1.9412841, -6.7265573, -1.9411418, -3.2238607, 3.2245178
8: -1.2038989, 0.5512741, -1.2038990, 0.5512744, -1.0630958, 1.0567801
9: -1.4500248, 0.4696424, -1.4500273, 0.4696712, -1.6184790, 1.6177723

Time for backsubstitution: 3.28 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 294
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 293

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2852542, upper bound: 0.2836215
time: 39.88 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853742, upper bound: 0.2853719
time: 40.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 83.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 83.98
Output dim: 1, lower bound: -0.2823243, upper bound: 0.2831441
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 83.98
Output dim: 1, lower bound: -0.2823318, upper bound: 0.2836936
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 83.98
Output dim: 1, lower bound: -0.2852542, upper bound: 0.2836215
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 83.98
Output dim: 1, lower bound: -0.2853742, upper bound: 0.2853719

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.7578030, 0.3366178, -0.7586390, 0.3381795, -0.4120388, 0.4114152
1: -0.7938410, 0.9824445, -0.7967278, 0.9846459, -1.2272916, 1.2286205
2: -2.7016125, -0.1800005, -2.6956029, -0.1811157, -1.6218297, 1.6167928
3: -7.1974473, -3.4054565, -7.1897354, -3.4073138, -1.4961256, 1.4903655
4: -6.0242414, -2.7201910, -6.0282650, -2.7180548, -1.6571307, 1.6579038
5: -7.5068150, -3.6225221, -7.4987545, -3.6237097, -1.6645706, 1.6594148
6: -10.5196428, -5.8251076, -10.5116386, -5.8243246, -1.4945337, 1.4858226
7: -6.7251205, -1.9416181, -6.7210975, -1.9437375, -3.2184210, 3.2171969
8: -1.1992610, 0.5421269, -1.2003965, 0.5446947, -1.0513830, 1.0501475
9: -1.4359188, 0.4685568, -1.4361974, 0.4641981, -1.5954285, 1.6017489

Time for backsubstitution: 3.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 293
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
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2642
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
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 296
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
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2742
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 422
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
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 3563
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 3260
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
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 3325
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3473
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 159
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
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2737
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3533
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 3223
type: A, layer: 1, pos: 3237
type: A, layer: 1, pos: 3582
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 2076
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
type: A, layer: 1, pos: 293

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2778872, upper bound: 0.2808550
time: 802.93 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2980

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2821256, upper bound: 0.2828649
time: 485.74 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2820502, upper bound: 0.2828665
time: 70.69 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.7581298, 0.3389753, -0.7608279, 0.3410535, -0.4118481, 0.4158630
1: -0.8014920, 0.9825004, -0.8062968, 0.9876438, -1.2375426, 1.2297632
2: -2.7040443, -0.1675059, -2.7017033, -0.1673834, -1.6321765, 1.6365066
3: -7.1978664, -3.3868654, -7.1962104, -3.3864708, -1.4929781, 1.5158429
4: -6.0258961, -2.7198460, -6.0291090, -2.7177825, -1.6612041, 1.6656485
5: -7.5071564, -3.6074216, -7.5028391, -3.6072741, -1.6611403, 1.6801226
6: -10.5196686, -5.7973256, -10.5205173, -5.7926025, -1.4904366, 1.5237129
7: -6.7294726, -1.9411906, -6.7251801, -1.9409502, -3.2253711, 3.2170734
8: -1.2020128, 0.5428886, -1.2035456, 0.5447886, -1.0547113, 1.0532091
9: -1.4419764, 0.4690988, -1.4438177, 0.4691428, -1.6109966, 1.6091878

Time for backsubstitution: 3.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 293
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
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2642
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
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2742
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 422
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
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2991
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
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2743
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 3325
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3473
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 159
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
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2737
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3533
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 3223
type: A, layer: 1, pos: 3237
type: A, layer: 1, pos: 3582
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 2076
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
type: A, layer: 1, pos: 293

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2778914, upper bound: 0.2814138
time: 549.68 seconds

## Relational analysis of IS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2980

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2821223, upper bound: 0.2834085
time: 239.61 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2820515, upper bound: 0.2834145
time: 518.33 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.7613303, 0.3485205, -0.7613030, 0.3480019, -0.4257965, 0.4153146
1: -0.8245134, 0.9888451, -0.8231860, 0.9887190, -1.2460146, 1.2593567
2: -2.7040071, -0.1664343, -2.7039139, -0.1663928, -1.6438022, 1.6428365
3: -7.1971459, -3.3864598, -7.1971245, -3.3863487, -1.5191468, 1.5199796
4: -6.0307159, -2.7090650, -6.0305691, -2.7095757, -1.6740279, 1.6681200
5: -7.5050287, -3.6084566, -7.5049095, -3.6081402, -1.6857090, 1.6850154
6: -10.5232048, -5.7910433, -10.5227404, -5.7911720, -1.5284002, 1.5319061
7: -6.7264695, -1.9413488, -6.7264338, -1.9412512, -3.2232914, 3.2240844
8: -1.2038479, 0.5509758, -1.2038176, 0.5507667, -1.0625122, 1.0563734
9: -1.4495556, 0.4695197, -1.4492278, 0.4694645, -1.6174109, 1.6166208

Time for backsubstitution: 3.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 422
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 3487
type: A, layer: 1, pos: 2967
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 293
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 422

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2846980, upper bound: 0.2836074
time: 49.57 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2852456, upper bound: 0.2836053
time: 217.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 270.33 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 270.33
Output dim: 1, lower bound: -0.2821256, upper bound: 0.2828649
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 270.33
Output dim: 1, lower bound: -0.2820502, upper bound: 0.2828665
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 270.33
Output dim: 1, lower bound: -0.2821223, upper bound: 0.2834085
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 270.33
Output dim: 1, lower bound: -0.2820515, upper bound: 0.2834145
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 270.33
Output dim: 1, lower bound: -0.2846980, upper bound: 0.2836074
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 270.33
Output dim: 1, lower bound: -0.2852456, upper bound: 0.2836053
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 270.33
Output dim: 1, lower bound: -0.2853742, upper bound: 0.2853719
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=4, k_mid=3, eps_mid=0.0117188, abs_max=1.2628133296966553
rel_dist={1: [-0.2853804858856417, 0.28537378025840754]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 294

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2554252, upper bound: 0.2563410
time: 34.64 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2580086, upper bound: 0.2580116
time: 37.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 72.72 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 72.72
Output dim: 1, lower bound: -0.2554252, upper bound: 0.2563410
IS_A2, status: Status.UNKNOWN, split count: 1, time: 72.72
Output dim: 1, lower bound: -0.2580086, upper bound: 0.2580116

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.7613775, 0.3492614, -0.7613797, 0.3492639, -0.4171711, 0.4060649
1: -0.8265103, 0.9890466, -0.8265124, 0.9890473, -1.2366219, 1.2514319
2: -2.7041450, -0.1663675, -2.7041476, -0.1663076, -1.6291888, 1.6282232
3: -7.1971817, -3.3863952, -7.1971817, -3.3862910, -1.4859736, 1.4867690
4: -6.0309362, -2.7083230, -6.0309401, -2.7083144, -1.6489787, 1.6426270
5: -7.5052214, -3.6084094, -7.5052271, -3.6081891, -1.6524959, 1.6517987
6: -10.5238705, -5.7908487, -10.5238724, -5.7908459, -1.4840108, 1.4879478
7: -6.7265453, -1.9412841, -6.7265539, -1.9411880, -3.1982427, 3.1989465
8: -1.2038989, 0.5512741, -1.2038989, 0.5512741, -1.0515143, 1.0451136
9: -1.4500248, 0.4696424, -1.4500263, 0.4696620, -1.6122073, 1.6115806

Time for backsubstitution: 2.95 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 293

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2578727, upper bound: 0.2567799
time: 238.60 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2580113, upper bound: 0.2580104
time: 89.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 330.77 seconds
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 330.77
Output dim: 1, lower bound: -0.2578727, upper bound: 0.2567799
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 330.77
Output dim: 1, lower bound: -0.2580113, upper bound: 0.2580104

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.7613147, 0.3491955, -0.7730955, 0.3492966, -0.4160337, 0.4176137
1: -0.8264534, 0.9890373, -0.8268545, 1.0214629, -1.2690065, 1.2508456
2: -2.7040722, -0.1705309, -2.7063127, -0.1712276, -1.6283121, 1.6296108
3: -7.1970887, -3.3892434, -7.1970162, -3.3891616, -1.4852557, 1.4856658
4: -6.0308614, -2.7083783, -6.0484715, -2.7077303, -1.6488755, 1.6614583
5: -7.5050087, -3.6154437, -7.5050945, -3.6168346, -1.6495337, 1.6552442
6: -10.5237169, -5.7909403, -10.5237036, -5.7596550, -1.5166438, 1.4871500
7: -6.7261438, -1.9484208, -6.7437963, -1.9499300, -3.1936460, 3.2154655
8: -1.2038578, 0.5511285, -1.2124596, 0.5520148, -1.0526764, 1.0521158
9: -1.4499525, 0.4692675, -1.4520164, 0.4693427, -1.6101367, 1.6144178

Time for backsubstitution: 2.93 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 422

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2576346, upper bound: 0.2579949
time: 330.73 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579997, upper bound: 0.2579966
time: 56.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 390.60 seconds
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 390.60
Output dim: 1, lower bound: -0.2576346, upper bound: 0.2579949
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 390.60
Output dim: 1, lower bound: -0.2579997, upper bound: 0.2579966

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.7591194, 0.3463213, -0.7727432, 0.3467932, -0.4114398, 0.4144288
1: -0.8168522, 0.9860348, -0.8187592, 1.0214005, -1.2598159, 1.2401810
2: -2.6979582, -0.1842734, -2.7037482, -0.1841089, -1.6080174, 1.6128523
3: -7.1906071, -3.4101324, -7.1965771, -3.4086308, -1.4585723, 1.4637262
4: -6.0300069, -2.7094278, -6.0467210, -2.7087541, -1.6434574, 1.6565465
5: -7.5009174, -3.6318986, -7.5047336, -3.6323953, -1.6277654, 1.6370702
6: -10.5148392, -5.8227549, -10.5236759, -5.7891712, -1.4769311, 1.4537694
7: -6.7219982, -1.9512112, -6.7391653, -1.9503839, -3.1891942, 3.2082276
8: -1.2006955, 0.5510275, -1.2095866, 0.5512130, -1.0484457, 1.0486817
9: -1.4423048, 0.4643220, -1.4456128, 0.4687642, -1.6019591, 1.6007138

Time for backsubstitution: 2.92 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2980

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2574295, upper bound: 0.2578425
time: 86.99 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2574241, upper bound: 0.2577904
time: 37.86 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7613078, 0.3491921, -0.7730892, 0.3492938, -0.4160259, 0.4142064
1: -0.8264169, 0.9890305, -0.8268223, 1.0214566, -1.2608519, 1.2508022
2: -2.7040601, -0.1705415, -2.7063017, -0.1712368, -1.6282907, 1.6232566
3: -7.1970844, -3.3892856, -7.1970115, -3.3891995, -1.4851763, 1.4602895
4: -6.0308552, -2.7091556, -6.0484657, -2.7084124, -1.6512494, 1.6606461
5: -7.5050011, -3.6154647, -7.5050859, -3.6168528, -1.6494174, 1.6334074
6: -10.5237179, -5.7910337, -10.5237026, -5.7597370, -1.5165297, 1.4492145
7: -6.7260885, -1.9484234, -6.7437458, -1.9499315, -3.1890328, 3.2153962
8: -1.2038441, 0.5511172, -1.2124479, 0.5520046, -1.0515424, 1.0520914
9: -1.4499177, 0.4692616, -1.4519864, 0.4693375, -1.6093785, 1.6166248

Time for backsubstitution: 2.96 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2980

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2577947, upper bound: 0.2578430
time: 39.91 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2577914, upper bound: 0.2577939
time: 35.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 78.35 seconds
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 78.35
Output dim: 1, lower bound: -0.2574295, upper bound: 0.2578425
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 78.35
Output dim: 1, lower bound: -0.2574241, upper bound: 0.2577904
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 78.35
Output dim: 1, lower bound: -0.2577947, upper bound: 0.2578430
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 78.35
Output dim: 1, lower bound: -0.2577914, upper bound: 0.2577939
Binary search (step 2): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.2514708042144775
rel_dist={1: [-0.2580117567799336, 0.2580088093270344]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 10152.00 seconds
