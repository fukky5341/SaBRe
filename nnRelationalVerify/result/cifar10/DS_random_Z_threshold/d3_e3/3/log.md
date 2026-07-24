## Execution arguments:
Dataset: Dataset.CIFAR10
Network: onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 7200 seconds
Split limit: 100
Threshold: 0.6107424462


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.3430281, -0.3028105, -2.3430281, -0.3028105, -1.4267485, 1.4267486)
1: (-0.3535357, 1.1736901, -0.3535357, 1.1736901, -1.4214689, 1.4214690)
2: (-4.3245311, -1.4554243, -4.3245311, -1.4554243, -2.3379021, 2.3379021)
3: (-5.5070839, -0.5651636, -5.5070839, -0.5651636, -2.9532700, 2.9532702)
4: (-5.4661922, -1.5440314, -5.4661922, -1.5440314, -3.3356152, 3.3356154)
5: (-6.7747240, -1.2658160, -6.7747240, -1.2658160, -3.4999781, 3.4999781)
6: (-4.5830259, 0.5625104, -4.5830259, 0.5625104, -2.9157641, 2.9157639)
7: (-6.5530591, -1.3395458, -6.5530591, -1.3395458, -2.8234594, 2.8234596)
8: (-1.5637501, 0.2362410, -1.5637501, 0.2362410, -1.5023448, 1.5023446)
9: (0.7303438, 1.2562890, 0.7303438, 1.2562890, -0.3038721, 0.3038721)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 5.73 + 191.56 = 197.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.6113538, upper bound: 0.6113483

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2996
type: RSZ, layer: 1, pos: 2037
type: RSZ, layer: 1, pos: 2399
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3592
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 3502
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 299
type: RSZ, layer: 1, pos: 303
type: RSZ, layer: 1, pos: 2998
type: RSZ, layer: 1, pos: 2979
type: RSZ, layer: 1, pos: 2336
type: RSZ, layer: 1, pos: 2173
type: RSZ, layer: 1, pos: 2351
type: RSZ, layer: 1, pos: 3171
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 2125
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 3020
type: RSZ, layer: 1, pos: 2244
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 3024
type: RSZ, layer: 1, pos: 2506
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3230
type: RSZ, layer: 1, pos: 2831
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 391
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 2952
type: RSZ, layer: 1, pos: 2038
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 342
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2299
type: RSZ, layer: 1, pos: 2978
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 3517
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 2406
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2248
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 3014
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2345
type: RSZ, layer: 1, pos: 3344
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 328
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 486
type: RSZ, layer: 1, pos: 2290
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3206
type: RSZ, layer: 1, pos: 2459
type: RSZ, layer: 1, pos: 3349
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3583
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2333
type: RSZ, layer: 1, pos: 3417
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2512
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 2283
type: RSZ, layer: 1, pos: 2174
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 307
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 3241
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2505
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 2995
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 3192
type: RSZ, layer: 1, pos: 3156
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 352
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 321
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 3579
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2637
type: RSZ, layer: 1, pos: 2275
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 3147
type: RSZ, layer: 1, pos: 476
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 3432
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2306
type: RSZ, layer: 1, pos: 3580
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2326
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 2469
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 351
type: RSZ, layer: 1, pos: 2286
type: RSZ, layer: 1, pos: 483
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2057
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3434
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2957
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 3015
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3183
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 355
type: RSZ, layer: 1, pos: 2243
type: RSZ, layer: 1, pos: 313
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 3576
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 2977
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 2309
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2292
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2475
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 3449
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3575
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2297
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 3593
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 2817
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3017
type: RSZ, layer: 1, pos: 2093
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 2602
type: RSZ, layer: 1, pos: 2349
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2323
type: RSZ, layer: 1, pos: 3599
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3282
type: RSZ, layer: 1, pos: 3428
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 411
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2334
type: RSZ, layer: 1, pos: 164
type: RSZ, layer: 1, pos: 2042
type: RSZ, layer: 1, pos: 2832
type: RSZ, layer: 1, pos: 2343
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2561
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3227
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2335
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 298
type: RSZ, layer: 1, pos: 2499
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 319
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2126
type: RSZ, layer: 1, pos: 3353
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2317
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 308
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 377
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2965
type: RSZ, layer: 1, pos: 2318
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 3016
type: RSZ, layer: 1, pos: 3499
type: RSZ, layer: 1, pos: 2098
type: RSZ, layer: 1, pos: 2491
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2124
type: RSZ, layer: 1, pos: 3429
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2583
type: RSZ, layer: 1, pos: 3491
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2647
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 3351
type: RSZ, layer: 1, pos: 2384
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 326
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3410
type: RSZ, layer: 1, pos: 2594
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2278
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 310
type: RSZ, layer: 1, pos: 2085
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2470
type: RSZ, layer: 1, pos: 2538
type: RSZ, layer: 1, pos: 2072
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 369
type: RSZ, layer: 1, pos: 2662
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3193
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 2277
type: RSZ, layer: 1, pos: 2327
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2668
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3578
type: RSZ, layer: 1, pos: 2344
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 3293
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3257
type: RSZ, layer: 1, pos: 2036
type: RSZ, layer: 1, pos: 3365
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2033
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2650
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3208
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 410
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 2997
type: RSZ, layer: 1, pos: 2959
type: RSZ, layer: 1, pos: 2361
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3465
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 507
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 402
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2490
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 2455
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2517
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 2082
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 2458
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2178
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3555
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2511
type: RSZ, layer: 1, pos: 2532
type: RSZ, layer: 1, pos: 2964
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 367
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 2975
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 407
type: RSZ, layer: 1, pos: 3469
type: RSZ, layer: 1, pos: 318
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 400
type: RSZ, layer: 1, pos: 2247
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 3509
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 3009
type: RSZ, layer: 1, pos: 2586
type: RSZ, layer: 1, pos: 2456
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 2833
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3466
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2403
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2953
type: RSZ, layer: 1, pos: 3506
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2982
type: RSZ, layer: 1, pos: 2121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2996

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.6107035, upper bound: 0.6107042
time: 761.67 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.6107035, upper bound: 0.6107017
time: 423.97 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1185.66 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1185.66
Output dim: 8, lower bound: -0.6107035, upper bound: 0.6107042
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1185.66
Output dim: 8, lower bound: -0.6107035, upper bound: 0.6107017

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 197.29 + 1185.66 = 1382.95 seconds
