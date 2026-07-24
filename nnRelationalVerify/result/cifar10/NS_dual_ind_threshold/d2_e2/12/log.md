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
execution time: IAR + RelationalAnalysis = 8.29 + 754.97 = 763.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2774611, upper bound: 0.2774589

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 415
type: A, layer: 1, pos: 416
type: A, layer: 1, pos: 414
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2574
type: A, layer: 1, pos: 387
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 3496
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 2133
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 3391
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 289
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 3394
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3408
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 356
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 324
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 3294
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3481
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2994
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3281
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 3563
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 3310
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 270
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 3307
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2276
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 3185
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2256
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 3169
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3148

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 415

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2770008, upper bound: 0.2758542
time: 586.64 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2774172, upper bound: 0.2774196
time: 25.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 611.71 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 611.71
Output dim: 3, lower bound: -0.2770008, upper bound: 0.2758542
NS_A2, status: Status.UNKNOWN, split count: 1, time: 611.71
Output dim: 3, lower bound: -0.2774172, upper bound: 0.2774196

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2.7287574, -1.2008851, -2.7288644, -1.2005154, -0.9963800, 0.9964938
1: -5.4304056, -3.0948486, -5.4305134, -3.0941658, -1.3751636, 1.3745545
2: -0.7815701, 0.0564898, -0.7819538, 0.0566521, -0.7850745, 0.7854860
3: -0.9325945, 0.1773849, -0.9327683, 0.1787720, -0.8590932, 0.8578650
4: -1.2312287, -0.2914755, -1.2317147, -0.2913588, -0.9398699, 0.9402392
5: -0.6777593, 0.4830872, -0.6778274, 0.4836491, -0.8534945, 0.8529661
6: -1.8021880, -0.0315303, -1.8023549, -0.0308863, -1.3123715, 1.3109560
7: -1.2390654, 0.7747509, -1.2396214, 0.7747977, -1.7529807, 1.7534783
8: -2.5801291, -0.1064262, -2.5801880, -0.1064212, -2.1120574, 2.1120737
9: -2.6965013, -0.4884224, -2.6966200, -0.4879932, -1.4542546, 1.4539199

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 416
type: B, layer: 1, pos: 414
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 394
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 415
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 387
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 288
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 256
type: B, layer: 1, pos: 2133
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 3391
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 289
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3548
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3394
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3408
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 356
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 324
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 3294
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2994
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3281
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 3563
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 3261
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3310
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 270
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 3307
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2276
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3185
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2256
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 3169
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3148

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 416

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2754967, upper bound: 0.2749931
time: 133.36 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2769786, upper bound: 0.2758340
time: 42.18 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.7283354, -1.1988587, -2.7284877, -1.1993093, -0.9980521, 1.0025790
1: -5.4369926, -3.0918434, -5.4308581, -3.0919614, -1.3839233, 1.3768889
2: -0.7850642, 0.0615635, -0.7831564, 0.0571792, -0.7879390, 0.7927454
3: -0.9495707, 0.1834122, -0.9333480, 0.1832737, -0.8809333, 0.8633654
4: -1.2348653, -0.2855654, -1.2331769, -0.2909788, -0.9438865, 0.9476116
5: -0.6863995, 0.4856570, -0.6780868, 0.4855520, -0.8645139, 0.8558146
6: -1.8178864, -0.0264074, -1.8029270, -0.0264156, -1.3355405, 1.3157964
7: -1.2419220, 0.7821819, -1.2417848, 0.7749459, -1.7547041, 1.7630689
8: -2.5797701, -0.1063845, -2.5794773, -0.1064057, -2.1126714, 2.1114125
9: -2.6994658, -0.4862256, -2.6969929, -0.4865384, -1.4587748, 1.4563899

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 416
type: B, layer: 1, pos: 414
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 394
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 415
type: B, layer: 1, pos: 387
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 288
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 256
type: B, layer: 1, pos: 2133
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 3391
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 289
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3548
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3394
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3408
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 356
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 324
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 3294
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2994
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3281
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 3563
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 3261
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 3310
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 270
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 3307
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2276
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3185
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2256
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 3169
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3148

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 416

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2754967, upper bound: 0.2765137
time: 187.42 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2773976, upper bound: 0.2773992
time: 68.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 262.01 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 262.01
Output dim: 3, lower bound: -0.2754967, upper bound: 0.2749931
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 262.01
Output dim: 3, lower bound: -0.2769786, upper bound: 0.2758340
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 262.01
Output dim: 3, lower bound: -0.2754967, upper bound: 0.2765137
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 262.01
Output dim: 3, lower bound: -0.2773976, upper bound: 0.2773992

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2.7279737, -1.2008961, -2.7278872, -1.2005298, -0.9970406, 0.9958470
1: -5.4304004, -3.0949183, -5.4305067, -3.0942426, -1.3727634, 1.3742688
2: -0.7815326, 0.0564843, -0.7819103, 0.0566453, -0.7856960, 0.7851298
3: -0.9325840, 0.1773221, -0.9327562, 0.1786940, -0.8564419, 0.8577909
4: -1.2311743, -0.2914794, -1.2316504, -0.2913639, -0.9398105, 0.9401709
5: -0.6777549, 0.4830012, -0.6778224, 0.4835529, -0.8532851, 0.8529300
6: -1.8021796, -0.0316219, -1.8023442, -0.0310004, -1.3068125, 1.3109071
7: -1.2390283, 0.7747486, -1.2395751, 0.7747953, -1.7529379, 1.7499722
8: -2.5798304, -0.1064270, -2.5798140, -0.1064215, -2.1117461, 2.1124907
9: -2.6964929, -0.4885283, -2.6966090, -0.4880877, -1.4541276, 1.4538941

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 414
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2574
type: A, layer: 1, pos: 387
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 416
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 3496
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 2133
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 3391
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 289
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 3394
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3408
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 356
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 324
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 3294
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3481
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2994
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3281
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 3563
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 3310
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 270
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 3307
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2276
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 3185
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2256
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 3169
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3148

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 414

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2753920, upper bound: 0.2753400
time: 423.17 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2768552, upper bound: 0.2756631
time: 267.45 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2.7281744, -1.2006884, -2.7283549, -1.2010818, -0.9950082, 0.9995196
1: -5.4364533, -3.0951147, -5.4282293, -3.0959969, -1.3801392, 1.3734934
2: -0.7831477, 0.0607113, -0.7803167, 0.0545122, -0.7830338, 0.7894887
3: -0.9486592, 0.1767826, -0.9274389, 0.1753995, -0.8718734, 0.8503861
4: -1.2327739, -0.2861643, -1.2301933, -0.2933710, -0.9394029, 0.9440289
5: -0.6859934, 0.4831516, -0.6764517, 0.4827130, -0.8608456, 0.8504574
6: -1.8170810, -0.0327077, -1.8001324, -0.0336157, -1.3266712, 1.3042436
7: -1.2383140, 0.7819581, -1.2372965, 0.7722359, -1.7484367, 1.7583703
8: -2.5794780, -0.1064222, -2.5789690, -0.1062841, -2.1124976, 2.1106460
9: -2.6989188, -0.4873466, -2.6960659, -0.4879494, -1.4570311, 1.4546568

Time for backsubstitution: 6.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 414
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2574
type: A, layer: 1, pos: 387
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 416
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 3496
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 2133
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 3391
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 289
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 3394
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3408
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 356
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 324
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 3294
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3481
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2994
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3281
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 3563
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3310
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 270
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 3307
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2276
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 3185
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2256
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 3169
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3148

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 414

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2753921, upper bound: 0.2745108
time: 399.83 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2757972, upper bound: 0.2764381
time: 58.05 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2.7276924, -1.1988697, -2.7276742, -1.1993232, -0.9985254, 1.0019473
1: -5.4369869, -3.0918915, -5.4308519, -3.0920219, -1.3813214, 1.3766168
2: -0.7850286, 0.0615585, -0.7831133, 0.0571724, -0.7885624, 0.7923896
3: -0.9495609, 0.1833494, -0.9333360, 0.1831957, -0.8782836, 0.8632933
4: -1.2348161, -0.2855692, -1.2331153, -0.2909837, -0.9438323, 0.9475461
5: -0.6863952, 0.4855708, -0.6780819, 0.4854557, -0.8643048, 0.8557786
6: -1.8178785, -0.0264938, -1.8029169, -0.0265248, -1.3299813, 1.3157479
7: -1.2418852, 0.7821798, -1.2417384, 0.7749429, -1.7546614, 1.7595627
8: -2.5794704, -0.1063850, -2.5791001, -0.1064060, -2.1123595, 2.1118269
9: -2.6994567, -0.4862828, -2.6969821, -0.4866095, -1.4586552, 1.4563725

Time for backsubstitution: 6.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 414
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2574
type: A, layer: 1, pos: 387
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 416
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 3496
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2133
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 3391
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 289
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 3394
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3408
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 356
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 324
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 3294
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 3481
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2994
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3281
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 3563
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 3310
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 270
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 3307
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2276
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 3185
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2256
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 3169
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3148

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 414

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2773244, upper bound: 0.2768035
time: 34.93 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2773245, upper bound: 0.2773257
time: 67.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 108.81 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 108.81
Output dim: 3, lower bound: -0.2753920, upper bound: 0.2753400
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 108.81
Output dim: 3, lower bound: -0.2768552, upper bound: 0.2756631
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 108.81
Output dim: 3, lower bound: -0.2753921, upper bound: 0.2745108
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 108.81
Output dim: 3, lower bound: -0.2757972, upper bound: 0.2764381
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 108.81
Output dim: 3, lower bound: -0.2773244, upper bound: 0.2768035
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 108.81
Output dim: 3, lower bound: -0.2773245, upper bound: 0.2773257

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.7295008, -1.2009410, -2.7278783, -1.2007186, -0.9969687, 0.9953260
1: -5.4280262, -3.0960054, -5.4284716, -3.0942426, -1.3725814, 1.3718213
2: -0.7825823, 0.0559906, -0.7818598, 0.0558762, -0.7858297, 0.7845086
3: -0.9402633, 0.1773731, -0.9327201, 0.1786289, -0.8643211, 0.8572369
4: -1.2323420, -0.2884729, -1.2315896, -0.2913654, -0.9409766, 0.9431167
5: -0.6835724, 0.4827222, -0.6778011, 0.4832560, -0.8588565, 0.8524361
6: -1.8057696, -0.0337805, -1.8023211, -0.0325696, -1.3137779, 1.3090069
7: -1.2377186, 0.7762692, -1.2385699, 0.7747948, -1.7526579, 1.7509038
8: -2.5801685, -0.1064029, -2.5798135, -0.1064234, -2.1120563, 2.1119165
9: -2.6972528, -0.4883351, -2.6966074, -0.4880996, -1.4553987, 1.4534293

Time for backsubstitution: 6.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 394
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 415
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 387
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 288
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 256
type: B, layer: 1, pos: 2133
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 3391
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 414
type: B, layer: 1, pos: 289
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3548
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3394
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3408
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 356
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 324
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 3294
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2994
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3281
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 3563
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 3261
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3310
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 270
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 3307
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2276
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3185
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2256
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 3169
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3148

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3064

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2767700, upper bound: 0.2754269
time: 192.70 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2767692, upper bound: 0.2755762
time: 233.13 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.7294369, -1.2007666, -2.7283468, -1.2012701, -0.9949383, 0.9990922
1: -5.4340801, -3.0962012, -5.4261951, -3.0959971, -1.3799593, 1.3710346
2: -0.7843361, 0.0602256, -0.7802901, 0.0537427, -0.7833343, 0.7889030
3: -0.9565787, 0.1768311, -0.9274218, 0.1753345, -0.8798949, 0.8498849
4: -1.2340492, -0.2831566, -1.2301478, -0.2933724, -0.9406768, 0.9469913
5: -0.6923047, 0.4831469, -0.6764452, 0.4826360, -0.8670235, 0.8501782
6: -1.8222589, -0.0338913, -1.8001204, -0.0346556, -1.3338704, 1.3024524
7: -1.2372656, 0.7834808, -1.2364399, 0.7722353, -1.7480562, 1.7595057
8: -2.5798244, -0.1063938, -2.5789683, -0.1062853, -2.1128156, 2.1100750
9: -2.6996813, -0.4871569, -2.6960654, -0.4879613, -1.4583054, 1.4541906

Time for backsubstitution: 6.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 394
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 415
type: B, layer: 1, pos: 387
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 288
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 256
type: B, layer: 1, pos: 2133
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 3391
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 414
type: B, layer: 1, pos: 289
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3548
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3394
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3408
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 356
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 324
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 3294
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 2994
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3281
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 3563
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 3261
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 3310
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 270
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 3307
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2276
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3185
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2256
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 3169
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3148

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3064

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2757109, upper bound: 0.2756740
time: 441.25 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2757103, upper bound: 0.2763499
time: 399.64 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 763.25 + 3604.60 = 4367.86 seconds
