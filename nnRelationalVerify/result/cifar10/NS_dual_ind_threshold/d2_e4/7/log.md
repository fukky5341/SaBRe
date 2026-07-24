## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 7)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.2382540075


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.5994030, 1.8214937, 0.5994030, 1.8214937, -0.9943012, 0.9943012)
1: (-3.3336077, 1.3654075, -3.3336077, 1.3654075, -4.5378466, 4.5378466)
2: (-2.1922996, -0.7156632, -2.1922996, -0.7156632, -0.9825721, 0.9825720)
3: (-1.9789040, -0.1579174, -1.9789040, -0.1579174, -1.2762668, 1.2762667)
4: (-1.6185975, -0.2122724, -1.6185975, -0.2122724, -0.9723870, 0.9723873)
5: (-3.1711562, -1.0485244, -3.1711562, -1.0485244, -1.4459666, 1.4459667)
6: (-4.2703099, -1.6667049, -4.2703099, -1.6667049, -2.0030415, 2.0030417)
7: (-3.3942292, 0.1676577, -3.3942292, 0.1676577, -2.0374856, 2.0374851)
8: (-1.2602749, 0.6203964, -1.2602749, 0.6203964, -1.4299788, 1.4299790)
9: (-2.6695652, 0.8552895, -2.6695652, 0.8552895, -3.4068832, 3.4068832)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 9.70 + 186.44 = 196.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2384919, upper bound: 0.2384930

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 3486
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 357
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 3505
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 325
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 3281
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 340
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 372
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 3237
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 3310
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2959
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3224

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3049

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2381825, upper bound: 0.2382846
time: 43.37 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2382810, upper bound: 0.2382805
time: 102.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 146.49 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 146.49
Output dim: 3, lower bound: -0.2381825, upper bound: 0.2382846
NS_A2, status: Status.UNKNOWN, split count: 1, time: 146.49
Output dim: 3, lower bound: -0.2382810, upper bound: 0.2382805

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.6001250, 1.8209097, 0.5995837, 1.8209665, -0.9931335, 0.9926718
1: -3.3323994, 1.3543916, -3.3330202, 1.3555694, -4.5268803, 4.5234556
2: -2.1910026, -0.7159139, -2.1911669, -0.7158936, -0.9778478, 0.9807720
3: -1.9710648, -0.1589405, -1.9718491, -0.1579372, -1.1925938, 1.2641478
4: -1.6153772, -0.2124915, -1.6157227, -0.2122819, -0.9592590, 0.9679654
5: -3.1627605, -1.0498860, -3.1636603, -1.0485350, -1.3672154, 1.4330008
6: -4.2687492, -1.6668918, -4.2689471, -1.6667216, -1.9998653, 2.0011144
7: -3.3840263, 0.1656258, -3.3849888, 0.1676443, -1.9451590, 2.0215087
8: -1.2561686, 0.6201243, -1.2599442, 0.6201490, -1.4258553, 1.4295585
9: -2.6657579, 0.8475633, -2.6690254, 0.8482714, -3.3961558, 3.3978021

Time for backsubstitution: 7.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 3486
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 2093
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3471
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2321
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 357
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3505
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 325
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 3281
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 340
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 312
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 372
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 3237
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 3310
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3224

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3065

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2380372, upper bound: 0.2380001
time: 215.29 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2380348, upper bound: 0.2381262
time: 33.49 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.5994701, 1.8213575, 0.5994651, 1.8213658, -0.9941204, 0.9941092
1: -3.3333824, 1.3607607, -3.3334024, 1.3610923, -4.5333781, 4.5330591
2: -2.1919317, -0.7157323, -2.1919672, -0.7157258, -0.9822075, 0.9822531
3: -1.9765862, -0.1579297, -1.9768091, -0.1579287, -1.2745866, 1.2747284
4: -1.6175902, -0.2122769, -1.6176832, -0.2122763, -0.9715303, 0.9716044
5: -3.1687641, -1.0485272, -3.1689951, -1.0485266, -1.4442401, 1.4443734
6: -4.2698307, -1.6667161, -4.2698755, -1.6667148, -2.0025597, 2.0026131
7: -3.3914313, 0.1676511, -3.3917024, 0.1676518, -2.0354950, 2.0357952
8: -1.2602117, 0.6203654, -1.2602158, 0.6203685, -1.4299260, 1.4299176
9: -2.6693833, 0.8530811, -2.6693988, 0.8532948, -3.4047968, 3.4045768

Time for backsubstitution: 7.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 3486
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 2093
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3471
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2321
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 357
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3505
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 325
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 3281
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 340
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 312
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 372
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 3237
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 3310
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3224

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3065

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2381259, upper bound: 0.2380008
time: 510.72 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2381172, upper bound: 0.2381269
time: 138.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 657.24 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 657.24
Output dim: 3, lower bound: -0.2380372, upper bound: 0.2380001
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 657.24
Output dim: 3, lower bound: -0.2380348, upper bound: 0.2381262
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 657.24
Output dim: 3, lower bound: -0.2381259, upper bound: 0.2380008
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 657.24
Output dim: 3, lower bound: -0.2381172, upper bound: 0.2381269

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 196.14 + 1060.26 = 1256.40 seconds
