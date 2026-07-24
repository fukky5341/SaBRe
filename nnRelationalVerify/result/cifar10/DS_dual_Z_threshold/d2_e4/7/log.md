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
execution time: IAR + RelationalAnalysis = 9.41 + 185.67 = 195.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2384919, upper bound: 0.2384930

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 592
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 3310
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 545
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 325
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3237
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 610
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 357
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 3488

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2382970, upper bound: 0.2382402
time: 86.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2382272, upper bound: 0.2383031
time: 243.37 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 329.77 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 329.77
Output dim: 3, lower bound: -0.2382970, upper bound: 0.2382402
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 329.77
Output dim: 3, lower bound: -0.2382272, upper bound: 0.2383031

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.5994030, 1.8214937, 0.5994030, 1.8214937, -0.9943005, 0.9943002
1: -3.3336077, 1.3654075, -3.3336077, 1.3654075, -4.5378456, 4.5378456
2: -2.1922996, -0.7156632, -2.1922996, -0.7156632, -0.9825708, 0.9825708
3: -1.9789040, -0.1579174, -1.9789040, -0.1579174, -1.2762669, 1.2762667
4: -1.6185975, -0.2122724, -1.6185975, -0.2122724, -0.9723803, 0.9723831
5: -3.1711562, -1.0485244, -3.1711562, -1.0485244, -1.4459659, 1.4459662
6: -4.2703099, -1.6667049, -4.2703099, -1.6667049, -2.0030293, 2.0030255
7: -3.3942292, 0.1676577, -3.3942292, 0.1676577, -2.0374842, 2.0374844
8: -1.2602749, 0.6203964, -1.2602749, 0.6203964, -1.4299693, 1.4299724
9: -2.6695652, 0.8552895, -2.6695652, 0.8552895, -3.4068811, 3.4068809

Time for backsubstitution: 7.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 592
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 3310
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 545
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 325
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3237
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 610
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 357
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 3051

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2381314, upper bound: 0.2381324
time: 296.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2381936, upper bound: 0.2380824
time: 312.77 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.5994030, 1.8214937, 0.5994030, 1.8214937, -0.9943002, 0.9943005
1: -3.3336077, 1.3654075, -3.3336077, 1.3654075, -4.5378456, 4.5378461
2: -2.1922996, -0.7156632, -2.1922996, -0.7156632, -0.9825708, 0.9825706
3: -1.9789040, -0.1579174, -1.9789040, -0.1579174, -1.2762669, 1.2762667
4: -1.6185975, -0.2122724, -1.6185975, -0.2122724, -0.9723832, 0.9723803
5: -3.1711562, -1.0485244, -3.1711562, -1.0485244, -1.4459661, 1.4459660
6: -4.2703099, -1.6667049, -4.2703099, -1.6667049, -2.0030255, 2.0030293
7: -3.3942292, 0.1676577, -3.3942292, 0.1676577, -2.0374846, 2.0374842
8: -1.2602749, 0.6203964, -1.2602749, 0.6203964, -1.4299722, 1.4299693
9: -2.6695652, 0.8552895, -2.6695652, 0.8552895, -3.4068811, 3.4068813

Time for backsubstitution: 7.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 592
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 3310
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 545
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 325
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3237
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 610
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 357
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 3051

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2380785, upper bound: 0.2381978
time: 76.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2381337, upper bound: 0.2381409
time: 351.26 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 434.95 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 434.95
Output dim: 3, lower bound: -0.2381314, upper bound: 0.2381324
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 434.95
Output dim: 3, lower bound: -0.2381936, upper bound: 0.2380824
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 434.95
Output dim: 3, lower bound: -0.2380785, upper bound: 0.2381978
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 434.95
Output dim: 3, lower bound: -0.2381337, upper bound: 0.2381409

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 195.08 + 1381.77 = 1576.85 seconds
