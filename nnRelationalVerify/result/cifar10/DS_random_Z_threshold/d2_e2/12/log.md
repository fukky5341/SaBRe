## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 12)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.2763512556


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.7293627, -1.1992732, -2.7293627, -1.1992732, -0.9996009, 0.9996009)
1: (-5.4308758, -3.0918431, -5.4308758, -3.0918431, -1.3779269, 1.3779271)
2: (-0.7832856, 0.0571969, -0.7832856, 0.0571969, -0.7873966, 0.7873966)
3: (-0.9333911, 0.1834605, -0.9333911, 0.1834605, -0.8645267, 0.8645267)
4: (-1.2333683, -0.2909660, -1.2333683, -0.2909660, -0.9424023, 0.9424023)
5: (-0.6780988, 0.4856851, -0.6780988, 0.4856851, -0.8559285, 0.8559285)
6: (-1.8029530, -0.0262471, -1.8029530, -0.0262471, -1.3184588, 1.3184587)
7: (-1.2419100, 0.7749556, -1.2419100, 0.7749556, -1.7560139, 1.7560139)
8: (-2.5803876, -0.1064014, -2.5803876, -0.1064014, -2.1124494, 2.1124492)
9: (-2.6970193, -0.4864569, -2.6970193, -0.4864569, -1.4562379, 1.4562379)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.50 + 755.25 = 762.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2774611, upper bound: 0.2774589

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 350
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 545
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3310
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2574
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 189
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 518
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 356
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3394
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3307
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2483

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 350

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2773337, upper bound: 0.2774590
time: 62.16 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2774572, upper bound: 0.2773325
time: 65.88 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 128.05 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 128.05
Output dim: 3, lower bound: -0.2773337, upper bound: 0.2774590
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 128.05
Output dim: 3, lower bound: -0.2774572, upper bound: 0.2773325

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.7293627, -1.1992732, -2.7293627, -1.1992732, -0.9930512, 0.9927204
1: -5.4308758, -3.0918431, -5.4308758, -3.0918431, -1.3551453, 1.3539379
2: -0.7832856, 0.0571969, -0.7832856, 0.0571969, -0.7864189, 0.7864633
3: -0.9333911, 0.1834605, -0.9333911, 0.1834605, -0.8654073, 0.8653678
4: -1.2333683, -0.2909660, -1.2333683, -0.2909660, -0.9424023, 0.9424023
5: -0.6780988, 0.4856851, -0.6780988, 0.4856851, -0.8583853, 0.8582339
6: -1.8029530, -0.0262471, -1.8029530, -0.0262471, -1.3167735, 1.3168215
7: -1.2419100, 0.7749556, -1.2419100, 0.7749556, -1.7549288, 1.7549822
8: -2.5803876, -0.1064014, -2.5803876, -0.1064014, -2.1104476, 2.1103752
9: -2.6970193, -0.4864569, -2.6970193, -0.4864569, -1.4476507, 1.4471912

Time for backsubstitution: 5.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 545
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3307
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 518
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 356
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3394
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 189
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2574
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 3310
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 534

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 716

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2772050, upper bound: 0.2773291
time: 92.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2772051, upper bound: 0.2773297
time: 207.68 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.7293627, -1.1992732, -2.7293627, -1.1992732, -0.9927204, 0.9930512
1: -5.4308758, -3.0918431, -5.4308758, -3.0918431, -1.3539380, 1.3551453
2: -0.7832856, 0.0571969, -0.7832856, 0.0571969, -0.7864633, 0.7864189
3: -0.9333911, 0.1834605, -0.9333911, 0.1834605, -0.8653678, 0.8654074
4: -1.2333683, -0.2909660, -1.2333683, -0.2909660, -0.9424023, 0.9424023
5: -0.6780988, 0.4856851, -0.6780988, 0.4856851, -0.8582338, 0.8583853
6: -1.8029530, -0.0262471, -1.8029530, -0.0262471, -1.3168215, 1.3167737
7: -1.2419100, 0.7749556, -1.2419100, 0.7749556, -1.7549820, 1.7549288
8: -2.5803876, -0.1064014, -2.5803876, -0.1064014, -2.1103754, 2.1104476
9: -2.6970193, -0.4864569, -2.6970193, -0.4864569, -1.4471912, 1.4476504

Time for backsubstitution: 5.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3394
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 189
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 545
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 3310
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 356
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3307
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 518
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2574
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3548

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2774448, upper bound: 0.2772252
time: 187.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2773509, upper bound: 0.2773187
time: 197.35 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 390.76 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 390.76
Output dim: 3, lower bound: -0.2772050, upper bound: 0.2773291
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 390.76
Output dim: 3, lower bound: -0.2772051, upper bound: 0.2773297
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 390.76
Output dim: 3, lower bound: -0.2774448, upper bound: 0.2772252
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 390.76
Output dim: 3, lower bound: -0.2773509, upper bound: 0.2773187

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7293627, -1.1992732, -2.7293627, -1.1992732, -0.9930490, 0.9927180
1: -5.4308758, -3.0918431, -5.4308758, -3.0918431, -1.3551365, 1.3539292
2: -0.7832856, 0.0571969, -0.7832856, 0.0571969, -0.7864158, 0.7864602
3: -0.9333911, 0.1834605, -0.9333911, 0.1834605, -0.8654069, 0.8653675
4: -1.2333683, -0.2909660, -1.2333683, -0.2909660, -0.9424023, 0.9424023
5: -0.6780988, 0.4856851, -0.6780988, 0.4856851, -0.8583841, 0.8582327
6: -1.8029530, -0.0262471, -1.8029530, -0.0262471, -1.3167723, 1.3168206
7: -1.2419100, 0.7749556, -1.2419100, 0.7749556, -1.7549255, 1.7549789
8: -2.5803876, -0.1064014, -2.5803876, -0.1064014, -2.1104465, 2.1103737
9: -2.6970193, -0.4864569, -2.6970193, -0.4864569, -1.4476311, 1.4471717

Time for backsubstitution: 5.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 3307
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 545
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2574
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 189
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 356
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 3310
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 518
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3394
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3065

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 57

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2771821, upper bound: 0.2773098
time: 63.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2771821, upper bound: 0.2773060
time: 428.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7293627, -1.1992732, -2.7293627, -1.1992732, -0.9930488, 0.9927182
1: -5.4308758, -3.0918431, -5.4308758, -3.0918431, -1.3551366, 1.3539290
2: -0.7832856, 0.0571969, -0.7832856, 0.0571969, -0.7864157, 0.7864602
3: -0.9333911, 0.1834605, -0.9333911, 0.1834605, -0.8654070, 0.8653675
4: -1.2333683, -0.2909660, -1.2333683, -0.2909660, -0.9424023, 0.9424023
5: -0.6780988, 0.4856851, -0.6780988, 0.4856851, -0.8583841, 0.8582327
6: -1.8029530, -0.0262471, -1.8029530, -0.0262471, -1.3167723, 1.3168205
7: -1.2419100, 0.7749556, -1.2419100, 0.7749556, -1.7549255, 1.7549788
8: -2.5803876, -0.1064014, -2.5803876, -0.1064014, -2.1104462, 2.1103740
9: -2.6970193, -0.4864569, -2.6970193, -0.4864569, -1.4476314, 1.4471717

Time for backsubstitution: 5.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3394
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 518
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 545
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2574
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 189
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 3310
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3307
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 356
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2087

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 852

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2771990, upper bound: 0.2773258
time: 554.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2772022, upper bound: 0.2773230
time: 35.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7293627, -1.1992732, -2.7293627, -1.1992732, -0.9925063, 0.9927681
1: -5.4308758, -3.0918431, -5.4308758, -3.0918431, -1.3534803, 1.3544866
2: -0.7832856, 0.0571969, -0.7832856, 0.0571969, -0.7864358, 0.7864112
3: -0.9333911, 0.1834605, -0.9333911, 0.1834605, -0.8651966, 0.8651101
4: -1.2333683, -0.2909660, -1.2333683, -0.2909660, -0.9424023, 0.9424023
5: -0.6780988, 0.4856851, -0.6780988, 0.4856851, -0.8581091, 0.8582180
6: -1.8029530, -0.0262471, -1.8029530, -0.0262471, -1.3163304, 1.3159637
7: -1.2419100, 0.7749556, -1.2419100, 0.7749556, -1.7547700, 1.7547902
8: -2.5803876, -0.1064014, -2.5803876, -0.1064014, -2.1100407, 2.1102502
9: -2.6970193, -0.4864569, -2.6970193, -0.4864569, -1.4468153, 1.4469166

Time for backsubstitution: 5.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 3307
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2574
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 518
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 545
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 3310
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 189
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3394
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 356
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2638

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3259

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2774389, upper bound: 0.2772240
time: 191.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2774429, upper bound: 0.2772204
time: 30.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7293627, -1.1992732, -2.7293627, -1.1992732, -0.9924374, 0.9928370
1: -5.4308758, -3.0918431, -5.4308758, -3.0918431, -1.3532791, 1.3546878
2: -0.7832856, 0.0571969, -0.7832856, 0.0571969, -0.7864557, 0.7863914
3: -0.9333911, 0.1834605, -0.9333911, 0.1834605, -0.8650705, 0.8652361
4: -1.2333683, -0.2909660, -1.2333683, -0.2909660, -0.9424023, 0.9424023
5: -0.6780988, 0.4856851, -0.6780988, 0.4856851, -0.8580665, 0.8582605
6: -1.8029530, -0.0262471, -1.8029530, -0.0262471, -1.3160117, 1.3162824
7: -1.2419100, 0.7749556, -1.2419100, 0.7749556, -1.7548432, 1.7547168
8: -2.5803876, -0.1064014, -2.5803876, -0.1064014, -2.1101778, 2.1101131
9: -2.6970193, -0.4864569, -2.6970193, -0.4864569, -1.4464574, 1.4472742

Time for backsubstitution: 5.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3310
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 189
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 518
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2574
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 324
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3307
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3394
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 545
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 356
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2085

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2773468, upper bound: 0.2773178
time: 305.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2773501, upper bound: 0.2773127
time: 435.66 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 746.66 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 746.66
Output dim: 3, lower bound: -0.2771821, upper bound: 0.2773098
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 746.66
Output dim: 3, lower bound: -0.2771821, upper bound: 0.2773060
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 746.66
Output dim: 3, lower bound: -0.2771990, upper bound: 0.2773258
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 746.66
Output dim: 3, lower bound: -0.2772022, upper bound: 0.2773230
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 746.66
Output dim: 3, lower bound: -0.2774389, upper bound: 0.2772240
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 746.66
Output dim: 3, lower bound: -0.2774429, upper bound: 0.2772204
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 746.66
Output dim: 3, lower bound: -0.2773468, upper bound: 0.2773178
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 746.66
Output dim: 3, lower bound: -0.2773501, upper bound: 0.2773127

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 762.75 + 2891.40 = 3654.15 seconds
