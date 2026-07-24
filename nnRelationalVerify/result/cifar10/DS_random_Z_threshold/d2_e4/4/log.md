## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.1331705961


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.1873665, 0.4837773, -1.1873665, 0.4837773, -1.4867804, 1.4867804)
1: (-2.1170709, 0.2502751, -2.1170709, 0.2502751, -2.2234113, 2.2234113)
2: (-3.4748113, -1.6202770, -3.4748113, -1.6202770, -0.9909365, 0.9909365)
3: (-4.3028584, -1.2227947, -4.3028584, -1.2227947, -2.3877850, 2.3877850)
4: (-4.6619167, -1.6836028, -4.6619167, -1.6836028, -1.7102453, 1.7102454)
5: (-5.1989450, -2.2466028, -5.1989450, -2.2466028, -2.3643110, 2.3643112)
6: (-7.0688610, -3.0764618, -7.0688610, -3.0764618, -2.3954346, 2.3954349)
7: (-3.9847794, -1.0733277, -3.9847794, -1.0733277, -2.3625772, 2.3625772)
8: (0.7784872, 1.2479708, 0.7784872, 1.2479708, -0.2810311, 0.2810311)
9: (-0.4310917, 1.0980161, -0.4310917, 1.0980161, -1.2326757, 1.2326756)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.89 + 84.22 = 92.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1333039, upper bound: 0.1333060

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 342
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 3237
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 406
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 409
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 325
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 356
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 357
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 407
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3487
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 1098
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2589

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1329406, upper bound: 0.1329450
time: 148.16 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1329414, upper bound: 0.1329459
time: 72.77 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 220.94 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 220.94
Output dim: 8, lower bound: -0.1329406, upper bound: 0.1329450
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 220.94
Output dim: 8, lower bound: -0.1329414, upper bound: 0.1329459

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 92.10 + 220.94 = 313.05 seconds
