## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 10)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.1506990501


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.2675078, -0.6231716, -3.2675078, -0.6231716, -2.4346266, 2.4346268)
1: (-7.8783927, -4.3791933, -7.8783927, -4.3791933, -2.4259844, 2.4259844)
2: (0.1568995, 1.1000149, 0.1568995, 1.1000149, -0.5766250, 0.5766250)
3: (-0.0722442, 0.6856132, -0.0722442, 0.6856132, -0.3505208, 0.3505208)
4: (-1.7857325, -0.6319879, -1.7857325, -0.6319879, -0.3295758, 0.3295758)
5: (0.0104022, 0.5207169, 0.0104022, 0.5207169, -0.1238502, 0.1238502)
6: (-2.2220764, 0.2063156, -2.2220764, 0.2063156, -0.8255010, 0.8255010)
7: (-1.1140480, 0.6597227, -1.1140480, 0.6597227, -1.5160441, 1.5160441)
8: (-6.3846807, -2.6664126, -6.3846807, -2.6664126, -2.6627262, 2.6627262)
9: (-6.2070456, -3.0964742, -6.2070456, -3.0964742, -1.9517030, 1.9517031)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.52 + 77.01 = 85.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.1508463, upper bound: 0.1508492

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3584
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 246
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3392
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 3409
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 430
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 629
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3268
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3394
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3163
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 664
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 630
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 3120
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 189
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3159
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2915

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1505631, upper bound: 0.1505607
time: 232.04 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1505558, upper bound: 0.1505623
time: 260.04 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 492.09 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 492.09
Output dim: 2, lower bound: -0.1505631, upper bound: 0.1505607
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 492.09
Output dim: 2, lower bound: -0.1505558, upper bound: 0.1505623

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 85.52 + 492.09 = 577.61 seconds
