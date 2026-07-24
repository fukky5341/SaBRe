## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 12)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.5631288075


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.5276713, -2.4423265, -4.5276713, -2.4423265, -1.5864455, 1.5864455)
1: (-6.1211724, -3.4930997, -6.1211724, -3.4930997, -2.1486473, 2.1486473)
2: (-0.7244447, -0.1349565, -0.7244447, -0.1349565, -0.5886118, 0.5886119)
3: (-1.5154128, -0.3556867, -1.5154128, -0.3556867, -0.8222893, 0.8222893)
4: (-0.0097116, 0.2296939, -0.0097116, 0.2296939, -0.2256894, 0.2256894)
5: (-1.2374356, -0.2871456, -1.2374356, -0.2871456, -0.6660860, 0.6660860)
6: (-0.6861967, 0.5823843, -0.6861967, 0.5823843, -1.2424762, 1.2424762)
7: (-1.0062286, 0.6276342, -1.0062286, 0.6276342, -1.4932142, 1.4932141)
8: (-4.7644367, -3.1794159, -4.7644367, -3.1794159, -1.2326608, 1.2326607)
9: (-4.2184620, -2.2445803, -4.2184620, -2.2445803, -1.3628056, 1.3628058)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.86 + 196.27 = 204.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.5636929, upper bound: 0.5636803

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 351
type: DSZ, layer: 1, pos: 421
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 365
type: DSZ, layer: 1, pos: 406
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 321
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3247
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3473
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2778
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3417
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3586

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 351

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5633718, upper bound: 0.5636810
time: 235.10 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5636921, upper bound: 0.5633593
time: 271.96 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 507.14 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 507.14
Output dim: 7, lower bound: -0.5633718, upper bound: 0.5636810
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 507.14
Output dim: 7, lower bound: -0.5636921, upper bound: 0.5633593

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.5276713, -2.4423265, -4.5276713, -2.4423265, -1.5859673, 1.5859427
1: -6.1211724, -3.4930997, -6.1211724, -3.4930997, -2.1472943, 2.1472363
2: -0.7244447, -0.1349565, -0.7244447, -0.1349565, -0.5883894, 0.5883808
3: -1.5154128, -0.3556867, -1.5154128, -0.3556867, -0.8204872, 0.8204452
4: -0.0097116, 0.2296939, -0.0097116, 0.2296939, -0.2256926, 0.2256927
5: -1.2374356, -0.2871456, -1.2374356, -0.2871456, -0.6638151, 0.6637799
6: -0.6861967, 0.5823843, -0.6861967, 0.5823843, -1.2424802, 1.2424788
7: -1.0062286, 0.6276342, -1.0062286, 0.6276342, -1.4928993, 1.4929097
8: -4.7644367, -3.1794159, -4.7644367, -3.1794159, -1.2325420, 1.2325341
9: -4.2184620, -2.2445803, -4.2184620, -2.2445803, -1.3615355, 1.3614817

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 421
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 365
type: DSZ, layer: 1, pos: 406
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 321
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3247
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3473
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2778
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3417
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3586

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 421

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5631765, upper bound: 0.5636809
time: 181.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5633531, upper bound: 0.5635022
time: 180.60 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.5276713, -2.4423265, -4.5276713, -2.4423265, -1.5859427, 1.5859675
1: -6.1211724, -3.4930997, -6.1211724, -3.4930997, -2.1472363, 2.1472945
2: -0.7244447, -0.1349565, -0.7244447, -0.1349565, -0.5883808, 0.5883894
3: -1.5154128, -0.3556867, -1.5154128, -0.3556867, -0.8204452, 0.8204871
4: -0.0097116, 0.2296939, -0.0097116, 0.2296939, -0.2256927, 0.2256926
5: -1.2374356, -0.2871456, -1.2374356, -0.2871456, -0.6637799, 0.6638151
6: -0.6861967, 0.5823843, -0.6861967, 0.5823843, -1.2424791, 1.2424800
7: -1.0062286, 0.6276342, -1.0062286, 0.6276342, -1.4929098, 1.4928992
8: -4.7644367, -3.1794159, -4.7644367, -3.1794159, -1.2325342, 1.2325420
9: -4.2184620, -2.2445803, -4.2184620, -2.2445803, -1.3614817, 1.3615354

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 421
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 365
type: DSZ, layer: 1, pos: 406
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 321
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3247
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3473
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2778
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3417
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3586

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 421

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5634977, upper bound: 0.5633569
time: 339.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5636757, upper bound: 0.5631845
time: 100.59 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 446.56 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 446.56
Output dim: 7, lower bound: -0.5631765, upper bound: 0.5636809
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 446.56
Output dim: 7, lower bound: -0.5633531, upper bound: 0.5635022
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 446.56
Output dim: 7, lower bound: -0.5634977, upper bound: 0.5633569
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 446.56
Output dim: 7, lower bound: -0.5636757, upper bound: 0.5631845

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.5276713, -2.4423265, -4.5276713, -2.4423265, -1.5861082, 1.5860759
1: -6.1211724, -3.4930997, -6.1211724, -3.4930997, -2.1468315, 2.1467581
2: -0.7244447, -0.1349565, -0.7244447, -0.1349565, -0.5884050, 0.5883955
3: -1.5154128, -0.3556867, -1.5154128, -0.3556867, -0.8199596, 0.8198997
4: -0.0097116, 0.2296939, -0.0097116, 0.2296939, -0.2256067, 0.2256094
5: -1.2374356, -0.2871456, -1.2374356, -0.2871456, -0.6639911, 0.6639439
6: -0.6861967, 0.5823843, -0.6861967, 0.5823843, -1.2424979, 1.2424967
7: -1.0062286, 0.6276342, -1.0062286, 0.6276342, -1.4928989, 1.4929096
8: -4.7644367, -3.1794159, -4.7644367, -3.1794159, -1.2325842, 1.2325747
9: -4.2184620, -2.2445803, -4.2184620, -2.2445803, -1.3612909, 1.3612289

Time for backsubstitution: 6.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 365
type: DSZ, layer: 1, pos: 406
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 321
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3247
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3473
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2778
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3417
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3586

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3129

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5630147, upper bound: 0.5636682
time: 43.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5631733, upper bound: 0.5635114
time: 59.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.5276713, -2.4423265, -4.5276713, -2.4423265, -1.5861003, 1.5860835
1: -6.1211724, -3.4930997, -6.1211724, -3.4930997, -2.1468160, 2.1467733
2: -0.7244447, -0.1349565, -0.7244447, -0.1349565, -0.5884041, 0.5883964
3: -1.5154128, -0.3556867, -1.5154128, -0.3556867, -0.8199416, 0.8199177
4: -0.0097116, 0.2296939, -0.0097116, 0.2296939, -0.2256094, 0.2256068
5: -1.2374356, -0.2871456, -1.2374356, -0.2871456, -0.6639793, 0.6639557
6: -0.6861967, 0.5823843, -0.6861967, 0.5823843, -1.2424979, 1.2424967
7: -1.0062286, 0.6276342, -1.0062286, 0.6276342, -1.4928992, 1.4929091
8: -4.7644367, -3.1794159, -4.7644367, -3.1794159, -1.2325826, 1.2325764
9: -4.2184620, -2.2445803, -4.2184620, -2.2445803, -1.3612828, 1.3612372

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 365
type: DSZ, layer: 1, pos: 406
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 321
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3247
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3473
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2778
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3417
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3586

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3129

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5631940, upper bound: 0.5634939
time: 129.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5633489, upper bound: 0.5633396
time: 249.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.5276713, -2.4423265, -4.5276713, -2.4423265, -1.5860834, 1.5861007
1: -6.1211724, -3.4930997, -6.1211724, -3.4930997, -2.1467736, 2.1468158
2: -0.7244447, -0.1349565, -0.7244447, -0.1349565, -0.5883964, 0.5884041
3: -1.5154128, -0.3556867, -1.5154128, -0.3556867, -0.8199177, 0.8199416
4: -0.0097116, 0.2296939, -0.0097116, 0.2296939, -0.2256068, 0.2256094
5: -1.2374356, -0.2871456, -1.2374356, -0.2871456, -0.6639558, 0.6639792
6: -0.6861967, 0.5823843, -0.6861967, 0.5823843, -1.2424967, 1.2424979
7: -1.0062286, 0.6276342, -1.0062286, 0.6276342, -1.4929094, 1.4928991
8: -4.7644367, -3.1794159, -4.7644367, -3.1794159, -1.2325764, 1.2325827
9: -4.2184620, -2.2445803, -4.2184620, -2.2445803, -1.3612370, 1.3612828

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 365
type: DSZ, layer: 1, pos: 406
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 321
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3247
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3473
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2778
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3417
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3586

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3129

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5633372, upper bound: 0.5631919
time: 696.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5634951, upper bound: 0.5631939
time: 170.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.5276713, -2.4423265, -4.5276713, -2.4423265, -1.5860758, 1.5861083
1: -6.1211724, -3.4930997, -6.1211724, -3.4930997, -2.1467578, 2.1468315
2: -0.7244447, -0.1349565, -0.7244447, -0.1349565, -0.5883955, 0.5884050
3: -1.5154128, -0.3556867, -1.5154128, -0.3556867, -0.8198997, 0.8199596
4: -0.0097116, 0.2296939, -0.0097116, 0.2296939, -0.2256095, 0.2256067
5: -1.2374356, -0.2871456, -1.2374356, -0.2871456, -0.6639440, 0.6639912
6: -0.6861967, 0.5823843, -0.6861967, 0.5823843, -1.2424967, 1.2424979
7: -1.0062286, 0.6276342, -1.0062286, 0.6276342, -1.4929097, 1.4928986
8: -4.7644367, -3.1794159, -4.7644367, -3.1794159, -1.2325747, 1.2325844
9: -4.2184620, -2.2445803, -4.2184620, -2.2445803, -1.3612289, 1.3612909

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 365
type: DSZ, layer: 1, pos: 406
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 321
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3247
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3473
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2778
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3417
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3586

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3129

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5635118, upper bound: 0.5631786
time: 44.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5636697, upper bound: 0.5630196
time: 150.95 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 202.13 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 202.13
Output dim: 7, lower bound: -0.5630147, upper bound: 0.5636682
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 202.13
Output dim: 7, lower bound: -0.5631733, upper bound: 0.5635114
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 202.13
Output dim: 7, lower bound: -0.5631940, upper bound: 0.5634939
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 202.13
Output dim: 7, lower bound: -0.5633489, upper bound: 0.5633396
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 202.13
Output dim: 7, lower bound: -0.5633372, upper bound: 0.5631919
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 202.13
Output dim: 7, lower bound: -0.5634951, upper bound: 0.5631939
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 202.13
Output dim: 7, lower bound: -0.5635118, upper bound: 0.5631786
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 202.13
Output dim: 7, lower bound: -0.5636697, upper bound: 0.5630196

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.5276713, -2.4423265, -4.5276713, -2.4423265, -1.5861073, 1.5860747
1: -6.1211724, -3.4930997, -6.1211724, -3.4930997, -2.1468282, 2.1467545
2: -0.7244447, -0.1349565, -0.7244447, -0.1349565, -0.5884039, 0.5883945
3: -1.5154128, -0.3556867, -1.5154128, -0.3556867, -0.8199589, 0.8198988
4: -0.0097116, 0.2296939, -0.0097116, 0.2296939, -0.2256067, 0.2256094
5: -1.2374356, -0.2871456, -1.2374356, -0.2871456, -0.6639901, 0.6639429
6: -0.6861967, 0.5823843, -0.6861967, 0.5823843, -1.2424977, 1.2424965
7: -1.0062286, 0.6276342, -1.0062286, 0.6276342, -1.4928982, 1.4929091
8: -4.7644367, -3.1794159, -4.7644367, -3.1794159, -1.2325819, 1.2325722
9: -4.2184620, -2.2445803, -4.2184620, -2.2445803, -1.3612895, 1.3612275

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 365
type: DSZ, layer: 1, pos: 406
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 321
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3247
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3473
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2778
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3417
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3586

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 365

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5627193, upper bound: 0.5632066
time: 179.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5630129, upper bound: 0.5633628
time: 376.91 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 562.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 562.72
Output dim: 7, lower bound: -0.5627193, upper bound: 0.5632066
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 562.72
Output dim: 7, lower bound: -0.5630129, upper bound: 0.5633628
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 562.72
Output dim: 7, lower bound: -0.5631733, upper bound: 0.5635114
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 562.72
Output dim: 7, lower bound: -0.5631940, upper bound: 0.5634939
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 562.72
Output dim: 7, lower bound: -0.5633489, upper bound: 0.5633396
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 562.72
Output dim: 7, lower bound: -0.5633372, upper bound: 0.5631919
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 562.72
Output dim: 7, lower bound: -0.5634951, upper bound: 0.5631939
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 562.72
Output dim: 7, lower bound: -0.5635118, upper bound: 0.5631786
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 562.72
Output dim: 7, lower bound: -0.5636697, upper bound: 0.5630196

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 204.12 + 3455.86 = 3659.98 seconds
