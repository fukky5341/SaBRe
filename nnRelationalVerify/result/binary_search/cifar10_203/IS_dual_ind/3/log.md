## Execution arguments:
Dataset: Dataset.CIFAR10
Network: onnx/cifar10_conv_exp.onnx
Epsilon: 0.03125
Initial delta epsilon: 8
Time budget: 18000 seconds
Threshold: 0.49434464943
Search space: {k/256.0 | k = 1, 2, ..., 8}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4500909, 1.4500908)
1: (-1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.9400406, 0.9400407)
2: (-3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.4278042, 1.4278041)
3: (-4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.6485567, 1.6485565)
4: (-5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.3103347, 2.3103347)
5: (-4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.6067874, 1.6067876)
6: (-8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.6424305, 1.6424305)
7: (-4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.2264836, 2.2264836)
8: (-0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.9023141, 0.9023140)
9: (-1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.2335304, 1.2335303)

## BASE Result
execution time: IAR + LP analysis = 5.93 + 133.34 = 139.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.6552173, upper bound: 0.6552176


# Binary Search by BASE starts (time budget: 17860.73 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=8, k_mid=4, eps_mid=0.0156250, abs_max=0.8798896074295044
rel_dist={1: [-0.5514959700660444, 0.5514987775895714]}

## Binary search (step 1) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=3, k_mid=2, eps_mid=0.0078125, abs_max=0.8498140573501587
rel_dist={1: [-0.49469755591069386, 0.49469970537319075]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=1, k_mid=1, eps_mid=0.0039062, abs_max=0.8347763419151306
rel_dist={1: [-0.4650599703525324, 0.4650619788368935]}

## Binary Search Result
Binary search time: 225.80 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Individual Split (IS_dual_ind) starts
Time budget: 17634.93 seconds

## Binary search (step 0) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2522

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786538, upper bound: 0.5785478
time: 15.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787125, upper bound: 0.5787132
time: 12.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 27.74 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 27.74
Output dim: 1, lower bound: -0.5786538, upper bound: 0.5785478
IS_A2, status: Status.UNKNOWN, split count: 1, time: 27.74
Output dim: 1, lower bound: -0.5787125, upper bound: 0.5787132

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.9573896, 0.7210121, -0.9576652, 0.7209611, -1.4297588, 1.4301732
1: -1.1015255, 0.6246668, -1.0994995, 0.6258909, -0.8895919, 0.8890654
2: -3.6656461, -1.6738957, -3.6657062, -1.6741275, -1.3863684, 1.3865920
3: -4.1976347, -0.7899210, -4.1973577, -0.7897239, -1.5560991, 1.5556242
4: -5.0255575, -1.6253605, -5.0260205, -1.6263218, -2.2398472, 2.2407784
5: -4.3330030, -1.0024294, -4.3328352, -1.0021437, -1.5125437, 1.5120316
6: -8.4153233, -4.5942554, -8.4143867, -4.5938921, -1.5223494, 1.5218492
7: -4.6257863, -1.2470965, -4.6258206, -1.2471628, -2.1598237, 2.1599448
8: -0.1768111, 0.7755800, -0.1768972, 0.7755066, -0.8944433, 0.8945795
9: -1.5134718, 0.1961916, -1.5125632, 0.1968295, -1.1938061, 1.1926198

Time for backsubstitution: 4.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2080

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785863, upper bound: 0.5779267
time: 8.12 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785868, upper bound: 0.5784834
time: 12.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.9584174, 0.7209653, -0.9584903, 0.7209661, -1.4303471, 1.4310529
1: -1.0996343, 0.6282543, -1.0996575, 0.6286402, -0.8945171, 0.8888938
2: -3.6658449, -1.6741171, -3.6658754, -1.6741157, -1.3866067, 1.3866484
3: -4.1973686, -0.7893600, -4.1973710, -0.7892942, -1.5563159, 1.5559208
4: -5.0270252, -1.6263089, -5.0271831, -1.6263068, -2.2405505, 2.2416706
5: -4.3328667, -1.0014926, -4.3328781, -1.0014310, -1.5131297, 1.5125358
6: -8.4143925, -4.5931339, -8.4143944, -4.5930200, -1.5236764, 1.5220076
7: -4.6258521, -1.2471557, -4.6259222, -1.2471540, -2.1601000, 2.1601591
8: -0.1771424, 0.7755325, -0.1771727, 0.7755364, -0.8948165, 0.8948799
9: -1.5127169, 0.1984509, -1.5127443, 0.1985998, -1.1960803, 1.1934549

Time for backsubstitution: 4.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2080

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786494, upper bound: 0.5780917
time: 73.56 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786480, upper bound: 0.5786488
time: 7.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 86.04 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 86.04
Output dim: 1, lower bound: -0.5785863, upper bound: 0.5779267
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 86.04
Output dim: 1, lower bound: -0.5785868, upper bound: 0.5784834
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 86.04
Output dim: 1, lower bound: -0.5786494, upper bound: 0.5780917
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 86.04
Output dim: 1, lower bound: -0.5786480, upper bound: 0.5786488

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.9573674, 0.7207391, -0.9573963, 0.7206089, -1.4293942, 1.4296836
1: -1.0994579, 0.6246611, -1.0967283, 0.6236808, -0.8865354, 0.8871088
2: -3.6656382, -1.6739049, -3.6656883, -1.6741484, -1.3863107, 1.3865606
3: -4.1973701, -0.7899241, -4.1970172, -0.7898973, -1.5558872, 1.5553365
4: -5.0255332, -1.6260241, -5.0255718, -1.6271546, -2.2391317, 2.2398751
5: -4.3328247, -1.0024352, -4.3325605, -1.0022500, -1.5121534, 1.5117013
6: -8.4144173, -4.5942612, -8.4132271, -4.5948300, -1.5218229, 1.5211257
7: -4.6257606, -1.2471753, -4.6253099, -1.2472677, -2.1596990, 2.1593781
8: -0.1767815, 0.7755736, -0.1768509, 0.7754776, -0.8943627, 0.8945001
9: -1.5124037, 0.1961894, -1.5110642, 0.1955185, -1.1921883, 1.1914432

Time for backsubstitution: 4.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5780893, upper bound: 0.5778687
time: 35.57 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783969, upper bound: 0.5778758
time: 76.84 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.9573762, 0.7208756, -0.9576476, 0.7207855, -1.4293500, 1.4300299
1: -1.1011913, 0.6246629, -1.0990686, 0.6258862, -0.8888750, 0.8861494
2: -3.6656423, -1.6739006, -3.6657021, -1.6741332, -1.3863611, 1.3865649
3: -4.1975884, -0.7899241, -4.1972995, -0.7897248, -1.5559998, 1.5553939
4: -5.0255432, -1.6256158, -5.0260019, -1.6266513, -2.2388048, 2.2404370
5: -4.3329039, -1.0024329, -4.3327079, -1.0021496, -1.5124381, 1.5116174
6: -8.4152012, -4.5942583, -8.4142303, -4.5938950, -1.5218289, 1.5213213
7: -4.6257730, -1.2471468, -4.6258044, -1.2472267, -2.1596889, 2.1598816
8: -0.1767937, 0.7755758, -0.1768749, 0.7755011, -0.8944005, 0.8945351
9: -1.5132935, 0.1961900, -1.5123339, 0.1968272, -1.1934516, 1.1909829

Time for backsubstitution: 4.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5780897, upper bound: 0.5782879
time: 15.82 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783963, upper bound: 0.5782949
time: 41.00 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.9583952, 0.7206921, -0.9582218, 0.7206137, -1.4299831, 1.4305632
1: -1.0975665, 0.6282486, -1.0968857, 0.6264297, -0.8914605, 0.8869365
2: -3.6658370, -1.6741260, -3.6658576, -1.6741371, -1.3865497, 1.3866162
3: -4.1971021, -0.7893625, -4.1970286, -0.7894686, -1.5561044, 1.5556326
4: -5.0270014, -1.6269727, -5.0267339, -1.6271384, -2.2398357, 2.2407672
5: -4.3326898, -1.0014998, -4.3326039, -1.0015377, -1.5127394, 1.5122058
6: -8.4134874, -4.5931392, -8.4132366, -4.5939593, -1.5231501, 1.5212839
7: -4.6258278, -1.2472351, -4.6254110, -1.2472603, -2.1599746, 2.1595931
8: -0.1771128, 0.7755260, -0.1771264, 0.7755075, -0.8947358, 0.8948007
9: -1.5116487, 0.1984487, -1.5112448, 0.1972890, -1.1944621, 1.1922781

Time for backsubstitution: 4.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5781504, upper bound: 0.5780338
time: 7.49 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784586, upper bound: 0.5780398
time: 48.25 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.9584037, 0.7208289, -0.9584729, 0.7207903, -1.4299390, 1.4309096
1: -1.0992997, 0.6282504, -1.0992264, 0.6286356, -0.8937999, 0.8859780
2: -3.6658416, -1.6741217, -3.6658716, -1.6741211, -1.3865993, 1.3866209
3: -4.1973219, -0.7893614, -4.1973104, -0.7892966, -1.5562164, 1.5556905
4: -5.0270119, -1.6265633, -5.0271645, -1.6266350, -2.2395082, 2.2413285
5: -4.3327680, -1.0014961, -4.3327494, -1.0014373, -1.5130241, 1.5121222
6: -8.4142704, -4.5931344, -8.4142399, -4.5930238, -1.5231564, 1.5214798
7: -4.6258402, -1.2472060, -4.6259069, -1.2472192, -2.1599650, 2.1600964
8: -0.1771249, 0.7755284, -0.1771504, 0.7755309, -0.8947740, 0.8948356
9: -1.5125387, 0.1984493, -1.5125145, 0.1985975, -1.1957252, 1.1918182

Time for backsubstitution: 4.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5781510, upper bound: 0.5784505
time: 7.70 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784577, upper bound: 0.5784551
time: 29.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 41.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 41.73
Output dim: 1, lower bound: -0.5780893, upper bound: 0.5778687
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 41.73
Output dim: 1, lower bound: -0.5783969, upper bound: 0.5778758
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 41.73
Output dim: 1, lower bound: -0.5780897, upper bound: 0.5782879
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 41.73
Output dim: 1, lower bound: -0.5783963, upper bound: 0.5782949
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 41.73
Output dim: 1, lower bound: -0.5781504, upper bound: 0.5780338
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 41.73
Output dim: 1, lower bound: -0.5784586, upper bound: 0.5780398
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 41.73
Output dim: 1, lower bound: -0.5781510, upper bound: 0.5784505
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 41.73
Output dim: 1, lower bound: -0.5784577, upper bound: 0.5784551

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.9576961, 0.7196849, -0.9573433, 0.7198860, -1.4287229, 1.4285190
1: -1.0923139, 0.6267971, -1.0915509, 0.6236688, -0.8771861, 0.8800394
2: -3.6658785, -1.6739451, -3.6656675, -1.6741865, -1.3863051, 1.3863782
3: -4.1968980, -0.7897366, -4.1966572, -0.7899079, -1.5551702, 1.5547497
4: -5.0258036, -1.6290214, -5.0254970, -1.6292127, -2.2365751, 2.2365055
5: -4.3325129, -1.0022055, -4.3323078, -1.0022699, -1.5116179, 1.5111780
6: -8.4122639, -4.5937338, -8.4116402, -4.5948448, -1.5189183, 1.5192857
7: -4.6265903, -1.2473278, -4.6252337, -1.2473798, -2.1598902, 2.1590476
8: -0.1767289, 0.7755845, -0.1767894, 0.7754591, -0.8942652, 0.8944659
9: -1.5094440, 0.1971204, -1.5087746, 0.1955117, -1.1889300, 1.1888106

Time for backsubstitution: 4.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776922, upper bound: 0.5774917
time: 18.89 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777118, upper bound: 0.5774905
time: 31.38 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.9573625, 0.7207018, -0.9573939, 0.7205892, -1.4293673, 1.4288142
1: -1.0990453, 0.6246601, -1.0965123, 0.6236801, -0.8756378, 0.8870932
2: -3.6656368, -1.6739545, -3.6656876, -1.6741745, -1.3863008, 1.3865182
3: -4.1973395, -0.7899262, -4.1970005, -0.7898983, -1.5549979, 1.5553092
4: -5.0255299, -1.6260725, -5.0255690, -1.6271796, -2.2390945, 2.2365823
5: -4.3327870, -1.0024370, -4.3325405, -1.0022501, -1.5113894, 1.5116805
6: -8.4143505, -4.5942621, -8.4131908, -4.5948315, -1.5186036, 1.5211208
7: -4.6257529, -1.2472060, -4.6253061, -1.2472848, -2.1596866, 2.1590817
8: -0.1767708, 0.7755730, -0.1768451, 0.7754774, -0.8944200, 0.8944824
9: -1.5122145, 0.1961882, -1.5109648, 0.1955180, -1.1881011, 1.1913999

Time for backsubstitution: 4.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5780008, upper bound: 0.5775021
time: 8.25 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5780229, upper bound: 0.5774999
time: 39.57 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.9577048, 0.7198910, -0.9575943, 0.7201523, -1.4286478, 1.4288673
1: -1.0940863, 0.6267991, -1.0939423, 0.6258744, -0.8805259, 0.8777545
2: -3.6658833, -1.6739414, -3.6656811, -1.6741695, -1.3863554, 1.3863828
3: -4.1971245, -0.7897367, -4.1969538, -0.7897369, -1.5553648, 1.5547078
4: -5.0258141, -1.6283872, -5.0259271, -1.6285670, -2.2361083, 2.2371459
5: -4.3326006, -1.0022036, -4.3324666, -1.0021695, -1.5119148, 1.5109653
6: -8.4131622, -4.5937295, -8.4127941, -4.5939097, -1.5193491, 1.5189717
7: -4.6266041, -1.2472802, -4.6257305, -1.2473179, -2.1598415, 2.1595936
8: -0.1767411, 0.7755874, -0.1768140, 0.7754835, -0.8943042, 0.8945020
9: -1.5103390, 0.1971211, -1.5100590, 0.1968200, -1.1905831, 1.1877228

Time for backsubstitution: 4.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776922, upper bound: 0.5779069
time: 9.84 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777128, upper bound: 0.5779080
time: 10.43 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.9573710, 0.7208384, -0.9576446, 0.7207661, -1.4293234, 1.4291611
1: -1.1007584, 0.6246617, -1.0988419, 0.6258855, -0.8803297, 0.8861342
2: -3.6656411, -1.6739500, -3.6657012, -1.6741586, -1.3863508, 1.3865223
3: -4.1975422, -0.7899241, -4.1972756, -0.7897254, -1.5553417, 1.5553668
4: -5.0255375, -1.6257536, -5.0259991, -1.6267232, -2.2387674, 2.2372303
5: -4.3328533, -1.0024352, -4.3326807, -1.0021507, -1.5118667, 1.5115967
6: -8.4150820, -4.5942593, -8.4141684, -4.5938954, -1.5195272, 1.5213159
7: -4.6257653, -1.2471769, -4.6258011, -1.2472434, -2.1596766, 2.1596751
8: -0.1767827, 0.7755753, -0.1768690, 0.7755009, -0.8944578, 0.8945174
9: -1.5130950, 0.1961890, -1.5122304, 0.1968267, -1.1904297, 1.1909397

Time for backsubstitution: 4.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5780013, upper bound: 0.5779169
time: 7.20 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5780202, upper bound: 0.5779154
time: 8.55 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.9587240, 0.7196383, -0.9581683, 0.7198910, -1.4293106, 1.4293990
1: -1.0904229, 0.6303847, -1.0917084, 0.6264181, -0.8821105, 0.8798676
2: -3.6660771, -1.6741664, -3.6658373, -1.6741743, -1.3865445, 1.3864346
3: -4.1966305, -0.7891741, -4.1966686, -0.7894779, -1.5553873, 1.5550468
4: -5.0272722, -1.6299701, -5.0266600, -1.6291964, -2.2372787, 2.2373977
5: -4.3323755, -1.0012693, -4.3323483, -1.0015591, -1.5122027, 1.5116825
6: -8.4113331, -4.5926123, -8.4116488, -4.5939746, -1.5202457, 1.5194440
7: -4.6266570, -1.2473872, -4.6253347, -1.2473706, -2.1601663, 2.1592622
8: -0.1770602, 0.7755369, -0.1770648, 0.7754890, -0.8946385, 0.8947662
9: -1.5086890, 0.1993801, -1.5089555, 0.1972824, -1.1912037, 1.1896461

Time for backsubstitution: 4.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777573, upper bound: 0.5776593
time: 8.22 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777773, upper bound: 0.5776579
time: 48.48 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.9583901, 0.7206553, -0.9582192, 0.7205943, -1.4299564, 1.4296942
1: -1.0971543, 0.6282474, -1.0966698, 0.6264291, -0.8805631, 0.8869213
2: -3.6658359, -1.6741755, -3.6658566, -1.6741631, -1.3865395, 1.3865743
3: -4.1970720, -0.7893625, -4.1970134, -0.7894681, -1.5552149, 1.5556059
4: -5.0269966, -1.6270208, -5.0267315, -1.6271642, -2.2397983, 2.2374742
5: -4.3326507, -1.0015014, -4.3325830, -1.0015380, -1.5119748, 1.5121849
6: -8.4134197, -4.5931396, -8.4132023, -4.5939608, -1.5199307, 1.5212789
7: -4.6258187, -1.2472653, -4.6254067, -1.2472758, -2.1599627, 2.1592960
8: -0.1771020, 0.7755255, -0.1771207, 0.7755072, -0.8947932, 0.8947828
9: -1.5114594, 0.1984481, -1.5111458, 0.1972885, -1.1903749, 1.1922348

Time for backsubstitution: 4.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5780665, upper bound: 0.5776635
time: 11.86 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5780840, upper bound: 0.5776673
time: 100.08 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.9587329, 0.7198443, -0.9584194, 0.7201571, -1.4292359, 1.4297469
1: -1.0921946, 0.6303866, -1.0941002, 0.6286234, -0.8854501, 0.8775834
2: -3.6660824, -1.6741624, -3.6658511, -1.6741583, -1.3865936, 1.3864396
3: -4.1968579, -0.7891736, -4.1969671, -0.7893059, -1.5555818, 1.5550041
4: -5.0272818, -1.6293349, -5.0270901, -1.6285502, -2.2368114, 2.2380381
5: -4.3324642, -1.0012674, -4.3325081, -1.0014589, -1.5125003, 1.5114697
6: -8.4122314, -4.5926080, -8.4128046, -4.5930386, -1.5206764, 1.5191302
7: -4.6266699, -1.2473390, -4.6258316, -1.2473087, -2.1601176, 2.1598082
8: -0.1770725, 0.7755399, -0.1770895, 0.7755133, -0.8946776, 0.8948022
9: -1.5095838, 0.1993807, -1.5102398, 0.1985903, -1.1928563, 1.1885587

Time for backsubstitution: 4.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777576, upper bound: 0.5780789
time: 84.55 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777770, upper bound: 0.5780779
time: 7.31 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.9583988, 0.7207918, -0.9584704, 0.7207711, -1.4299119, 1.4300411
1: -1.0988667, 0.6282495, -1.0990000, 0.6286348, -0.8852552, 0.8859625
2: -3.6658401, -1.6741711, -3.6658709, -1.6741471, -1.3865894, 1.3865788
3: -4.1972771, -0.7893624, -4.1972871, -0.7892969, -1.5555587, 1.5556631
4: -5.0270052, -1.6267006, -5.0271616, -1.6267078, -2.2394710, 2.2381222
5: -4.3327174, -1.0014988, -4.3327227, -1.0014378, -1.5124524, 1.5121011
6: -8.4141512, -4.5931368, -8.4141760, -4.5930252, -1.5208540, 1.5214746
7: -4.6258326, -1.2472367, -4.6259022, -1.2472351, -2.1599526, 2.1598897
8: -0.1771140, 0.7755278, -0.1771446, 0.7755307, -0.8948311, 0.8948178
9: -1.5123407, 0.1984489, -1.5124110, 0.1985973, -1.1927036, 1.1917751

Time for backsubstitution: 4.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5780667, upper bound: 0.5780817
time: 7.12 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5780863, upper bound: 0.5780834
time: 83.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 95.43 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5776922, upper bound: 0.5774917
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5777118, upper bound: 0.5774905
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5780008, upper bound: 0.5775021
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5780229, upper bound: 0.5774999
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5776922, upper bound: 0.5779069
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5777128, upper bound: 0.5779080
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5780013, upper bound: 0.5779169
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5780202, upper bound: 0.5779154
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5777573, upper bound: 0.5776593
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5777773, upper bound: 0.5776579
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5780665, upper bound: 0.5776635
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5780840, upper bound: 0.5776673
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5777576, upper bound: 0.5780789
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5777770, upper bound: 0.5780779
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5780667, upper bound: 0.5780817
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 95.43
Output dim: 1, lower bound: -0.5780863, upper bound: 0.5780834

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.9574727, 0.7196805, -0.9570407, 0.7198802, -1.4284511, 1.4281754
1: -1.0923032, 0.6259524, -1.0915364, 0.6225195, -0.8760224, 0.8791652
2: -3.6657856, -1.6739509, -3.6655436, -1.6741935, -1.3861978, 1.3862441
3: -4.1968899, -0.7898934, -4.1966443, -0.7901206, -1.5550034, 1.5546210
4: -5.0254831, -1.6290321, -5.0250616, -1.6292264, -2.2362900, 2.2361300
5: -4.3324871, -1.0023996, -4.3322716, -1.0025337, -1.5113208, 1.5109364
6: -8.4122581, -4.5940952, -8.4116316, -4.5953369, -1.5185285, 1.5189923
7: -4.6265073, -1.2473297, -4.6251202, -1.2473830, -2.1597922, 2.1589184
8: -0.1766702, 0.7755648, -0.1767094, 0.7754327, -0.8941593, 0.8943446
9: -1.5094230, 0.1969139, -1.5087461, 0.1952308, -1.1887007, 1.1886305

Time for backsubstitution: 4.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775897, upper bound: 0.5773421
time: 40.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775878, upper bound: 0.5773854
time: 48.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.9572957, 0.7196810, -0.9568874, 0.7200965, -1.4285975, 1.4281688
1: -1.0923003, 0.6250758, -1.0934294, 0.6214184, -0.8758857, 0.8824407
2: -3.6657414, -1.6739490, -3.6655154, -1.6739519, -1.3863294, 1.3862321
3: -4.1968861, -0.7900727, -4.1970954, -0.7903423, -1.5550232, 1.5548860
4: -5.0253420, -1.6290362, -5.0249281, -1.6286914, -2.2363434, 2.2360976
5: -4.3324585, -1.0026147, -4.3329506, -1.0027946, -1.5112544, 1.5115812
6: -8.4122515, -4.5944681, -8.4122076, -4.5958037, -1.5184636, 1.5192361
7: -4.6262903, -1.2473323, -4.6250930, -1.2473843, -2.1595933, 2.1589146
8: -0.1766010, 0.7755587, -0.1766489, 0.7758301, -0.8945333, 0.8942580
9: -1.5094181, 0.1964585, -1.5097564, 0.1946586, -1.1887636, 1.1894195

Time for backsubstitution: 4.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776083, upper bound: 0.5773426
time: 21.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776088, upper bound: 0.5773875
time: 42.30 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.9571390, 0.7206974, -0.9570913, 0.7205830, -1.4290965, 1.4284713
1: -1.0990342, 0.6238152, -1.0964975, 0.6225308, -0.8744739, 0.8862193
2: -3.6655438, -1.6739600, -3.6655629, -1.6741821, -1.3861935, 1.3863842
3: -4.1973310, -0.7900821, -4.1969881, -0.7901101, -1.5548311, 1.5551801
4: -5.0252066, -1.6260836, -5.0251322, -1.6271944, -2.2388093, 2.2362068
5: -4.3327608, -1.0026313, -4.3325047, -1.0025148, -1.5110924, 1.5114391
6: -8.4143448, -4.5946236, -8.4131842, -4.5953226, -1.5182140, 1.5208272
7: -4.6256695, -1.2472078, -4.6251926, -1.2472866, -2.1595891, 2.1589525
8: -0.1767120, 0.7755537, -0.1767651, 0.7754511, -0.8943143, 0.8943610
9: -1.5121930, 0.1959818, -1.5109360, 0.1952372, -1.1878712, 1.1912196

Time for backsubstitution: 4.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778975, upper bound: 0.5773463
time: 8.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778980, upper bound: 0.5773942
time: 29.98 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.9569616, 0.7206979, -0.9569383, 0.7207997, -1.4292427, 1.4284637
1: -1.0990317, 0.6229380, -1.0983908, 0.6214294, -0.8743374, 0.8894951
2: -3.6654999, -1.6739579, -3.6655359, -1.6739409, -1.3863246, 1.3863721
3: -4.1973276, -0.7902619, -4.1974387, -0.7903337, -1.5548506, 1.5554445
4: -5.0250664, -1.6260867, -5.0250001, -1.6266600, -2.2388630, 2.2361739
5: -4.3327327, -1.0028447, -4.3331881, -1.0027736, -1.5110258, 1.5120838
6: -8.4143372, -4.5949960, -8.4137592, -4.5957904, -1.5181487, 1.5210713
7: -4.6254544, -1.2472100, -4.6251655, -1.2472891, -2.1593895, 2.1589484
8: -0.1766429, 0.7755474, -0.1767046, 0.7758485, -0.8946882, 0.8942744
9: -1.5121878, 0.1955266, -1.5119464, 0.1946651, -1.1879340, 1.1920090

Time for backsubstitution: 4.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779183, upper bound: 0.5773465
time: 14.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779175, upper bound: 0.5773957
time: 38.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.9574814, 0.7198868, -0.9572920, 0.7201460, -1.4283764, 1.4285238
1: -1.0940758, 0.6259544, -1.0939276, 0.6247251, -0.8793626, 0.8768804
2: -3.6657898, -1.6739466, -3.6655579, -1.6741774, -1.3862480, 1.3862485
3: -4.1971154, -0.7898921, -4.1969414, -0.7899482, -1.5551986, 1.5545788
4: -5.0254927, -1.6283977, -5.0254917, -1.6285808, -2.2358234, 2.2367706
5: -4.3325739, -1.0023979, -4.3324304, -1.0024335, -1.5116181, 1.5107238
6: -8.4131575, -4.5940909, -8.4127874, -4.5944004, -1.5189593, 1.5186783
7: -4.6265197, -1.2472817, -4.6256170, -1.2473203, -2.1597438, 2.1594641
8: -0.1766825, 0.7755680, -0.1767339, 0.7754571, -0.8941985, 0.8943808
9: -1.5103180, 0.1969145, -1.5100305, 0.1965393, -1.1903534, 1.1875430

Time for backsubstitution: 4.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775870, upper bound: 0.5777592
time: 10.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775872, upper bound: 0.5778071
time: 41.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.9573040, 0.7198870, -0.9571387, 0.7203627, -1.4285231, 1.4285166
1: -1.0940728, 0.6250777, -1.0958216, 0.6236237, -0.8792259, 0.8801563
2: -3.6657455, -1.6739451, -3.6655302, -1.6739361, -1.3863795, 1.3862368
3: -4.1971121, -0.7900718, -4.1973929, -0.7901709, -1.5552180, 1.5548434
4: -5.0253506, -1.6284009, -5.0253587, -1.6280458, -2.2358770, 2.2367382
5: -4.3325472, -1.0026120, -4.3331118, -1.0026934, -1.5115516, 1.5113685
6: -8.4131489, -4.5944643, -8.4133644, -4.5948682, -1.5188944, 1.5189226
7: -4.6263046, -1.2472841, -4.6255894, -1.2473222, -2.1595445, 2.1594596
8: -0.1766135, 0.7755617, -0.1766734, 0.7758547, -0.8945723, 0.8942941
9: -1.5103129, 0.1964594, -1.5110414, 0.1959671, -1.1904167, 1.1883324

Time for backsubstitution: 4.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776060, upper bound: 0.5777576
time: 13.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776089, upper bound: 0.5778032
time: 37.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.9571476, 0.7208340, -0.9573425, 0.7207601, -1.4290524, 1.4288180
1: -1.1007477, 0.6238168, -1.0988276, 0.6247363, -0.8791661, 0.8852602
2: -3.6655488, -1.6739557, -3.6655774, -1.6741662, -1.3862436, 1.3863890
3: -4.1975350, -0.7900810, -4.1972628, -0.7899379, -1.5551748, 1.5552375
4: -5.0252166, -1.6257634, -5.0255628, -1.6267383, -2.2384822, 2.2368550
5: -4.3328285, -1.0026295, -4.3326473, -1.0024139, -1.5115697, 1.5113554
6: -8.4150772, -4.5946207, -8.4141607, -4.5943871, -1.5191371, 1.5210226
7: -4.6256814, -1.2471786, -4.6256871, -1.2472465, -2.1595793, 2.1595459
8: -0.1767238, 0.7755558, -0.1767891, 0.7754745, -0.8943521, 0.8943961
9: -1.5130739, 0.1959824, -1.5122015, 0.1965457, -1.1902001, 1.1907599

Time for backsubstitution: 4.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778969, upper bound: 0.5777654
time: 11.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778983, upper bound: 0.5778137
time: 16.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.9569700, 0.7208344, -0.9571894, 0.7209764, -1.4291985, 1.4288106
1: -1.1007446, 0.6229402, -1.1007211, 0.6236349, -0.8790298, 0.8885358
2: -3.6655040, -1.6739541, -3.6655493, -1.6739252, -1.3863747, 1.3863765
3: -4.1975327, -0.7902609, -4.1977124, -0.7901617, -1.5551946, 1.5555029
4: -5.0250745, -1.6257675, -5.0254297, -1.6262019, -2.2385361, 2.2368214
5: -4.3327999, -1.0028439, -4.3333268, -1.0026731, -1.5115030, 1.5119996
6: -8.4150696, -4.5949922, -8.4147377, -4.5948544, -1.5190724, 1.5212669
7: -4.6254659, -1.2471813, -4.6256604, -1.2472479, -2.1593795, 2.1595416
8: -0.1766548, 0.7755495, -0.1767288, 0.7758721, -0.8947260, 0.8943094
9: -1.5130694, 0.1955272, -1.5132117, 0.1959737, -1.1902632, 1.1915486

Time for backsubstitution: 4.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779156, upper bound: 0.5777658
time: 63.51 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779179, upper bound: 0.5778106
time: 15.21 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.9585007, 0.7196338, -0.9578664, 0.7198852, -1.4290401, 1.4290555
1: -1.0904119, 0.6295400, -1.0916938, 0.6252686, -0.8809475, 0.8789940
2: -3.6659842, -1.6741724, -3.6657131, -1.6741817, -1.3864363, 1.3863007
3: -4.1966219, -0.7893302, -4.1966558, -0.7896904, -1.5552201, 1.5549179
4: -5.0269518, -1.6299806, -5.0262241, -1.6292101, -2.2369943, 2.2370229
5: -4.3323498, -1.0014632, -4.3323135, -1.0018213, -1.5119052, 1.5114411
6: -8.4113274, -4.5929723, -8.4116411, -4.5944662, -1.5198559, 1.5191505
7: -4.6265726, -1.2473894, -4.6252217, -1.2473742, -2.1600683, 2.1591332
8: -0.1770014, 0.7755174, -0.1769848, 0.7754626, -0.8945326, 0.8946451
9: -1.5086677, 0.1991730, -1.5089267, 0.1970010, -1.1909736, 1.1894665

Time for backsubstitution: 4.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776538, upper bound: 0.5775105
time: 134.28 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776511, upper bound: 0.5775586
time: 24.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.9583232, 0.7196344, -0.9577132, 0.7201018, -1.4291863, 1.4290490
1: -1.0904095, 0.6286630, -1.0935874, 0.6241669, -0.8808123, 0.8822693
2: -3.6659405, -1.6741710, -3.6656854, -1.6739403, -1.3865680, 1.3862883
3: -4.1966200, -0.7895097, -4.1971078, -0.7899148, -1.5552398, 1.5551822
4: -5.0268097, -1.6299844, -5.0260901, -1.6286746, -2.2370467, 2.2369895
5: -4.3323221, -1.0016775, -4.3329935, -1.0020807, -1.5118382, 1.5120860
6: -8.4113188, -4.5933433, -8.4122171, -4.5949326, -1.5197911, 1.5193950
7: -4.6263571, -1.2473922, -4.6251941, -1.2473764, -2.1598692, 2.1591287
8: -0.1769323, 0.7755111, -0.1769244, 0.7758600, -0.8949069, 0.8945587
9: -1.5086633, 0.1987180, -1.5099375, 0.1964290, -1.1910374, 1.1902559

Time for backsubstitution: 4.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776724, upper bound: 0.5775113
time: 20.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776721, upper bound: 0.5775571
time: 95.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.9581671, 0.7206509, -0.9579175, 0.7205882, -1.4296850, 1.4293506
1: -1.0971431, 0.6274024, -1.0966551, 0.6252797, -0.8794006, 0.8860475
2: -3.6657426, -1.6741810, -3.6657324, -1.6741705, -1.3864324, 1.3864408
3: -4.1970625, -0.7895188, -4.1970010, -0.7896807, -1.5550481, 1.5554769
4: -5.0266752, -1.6270316, -5.0262947, -1.6271778, -2.2395129, 2.2370992
5: -4.3326240, -1.0016940, -4.3325477, -1.0018016, -1.5116775, 1.5119438
6: -8.4134130, -4.5935011, -8.4131918, -4.5944514, -1.5195410, 1.5209856
7: -4.6257358, -1.2472667, -4.6252933, -1.2472795, -2.1598651, 2.1591673
8: -0.1770433, 0.7755062, -0.1770406, 0.7754810, -0.8946878, 0.8946615
9: -1.5114383, 0.1982409, -1.5111171, 0.1970076, -1.1901453, 1.1920547

Time for backsubstitution: 4.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779622, upper bound: 0.5775142
time: 13.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779622, upper bound: 0.5775599
time: 54.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.9579895, 0.7206511, -0.9577640, 0.7208048, -1.4298317, 1.4293439
1: -1.0971406, 0.6265259, -1.0985485, 0.6241784, -0.8792651, 0.8893232
2: -3.6656985, -1.6741798, -3.6657050, -1.6739292, -1.3865633, 1.3864281
3: -4.1970596, -0.7896984, -4.1974516, -0.7899046, -1.5550677, 1.5557414
4: -5.0265336, -1.6270347, -5.0261631, -1.6266429, -2.2395663, 2.2370667
5: -4.3325949, -1.0019084, -4.3332272, -1.0020622, -1.5116100, 1.5125880
6: -8.4134054, -4.5938745, -8.4137697, -4.5949192, -1.5194762, 1.5212300
7: -4.6255207, -1.2472696, -4.6252666, -1.2472806, -2.1596656, 2.1591625
8: -0.1769741, 0.7754999, -0.1769800, 0.7758783, -0.8950616, 0.8945752
9: -1.5114341, 0.1977857, -1.5121269, 0.1964354, -1.1902083, 1.1928446

Time for backsubstitution: 4.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779819, upper bound: 0.5775134
time: 7.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779827, upper bound: 0.5775628
time: 86.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.9585096, 0.7198398, -0.9581176, 0.7201511, -1.4289656, 1.4294035
1: -1.0921836, 0.6295419, -1.0940852, 0.6274742, -0.8842872, 0.8767095
2: -3.6659887, -1.6741673, -3.6657279, -1.6741652, -1.3864865, 1.3863053
3: -4.1968479, -0.7893294, -4.1969528, -0.7895185, -1.5554152, 1.5548756
4: -5.0269608, -1.6293476, -5.0266538, -1.6285650, -2.2365270, 2.2376633
5: -4.3324380, -1.0014613, -4.3324728, -1.0017205, -1.5122023, 1.5112287
6: -8.4122248, -4.5929699, -8.4127951, -4.5935297, -1.5202869, 1.5188371
7: -4.6265860, -1.2473420, -4.6257181, -1.2473121, -2.1600199, 2.1596785
8: -0.1770137, 0.7755205, -0.1770093, 0.7754869, -0.8945715, 0.8946812
9: -1.5095625, 0.1991739, -1.5102113, 0.1983093, -1.1926265, 1.1883787

Time for backsubstitution: 4.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776525, upper bound: 0.5779264
time: 15.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776525, upper bound: 0.5779700
time: 16.43 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.9583318, 0.7198403, -0.9579642, 0.7203676, -1.4291115, 1.4293963
1: -1.0921812, 0.6286651, -1.0959793, 0.6263727, -0.8841518, 0.8799849
2: -3.6659455, -1.6741660, -3.6656985, -1.6739247, -1.3866179, 1.3862927
3: -4.1968460, -0.7895101, -4.1974039, -0.7897426, -1.5554345, 1.5551395
4: -5.0268183, -1.6293495, -5.0265207, -1.6280290, -2.2365806, 2.2376301
5: -4.3324099, -1.0016754, -4.3331532, -1.0019805, -1.5121353, 1.5118731
6: -8.4122190, -4.5933418, -8.4133711, -4.5939980, -1.5202222, 1.5190814
7: -4.6263714, -1.2473441, -4.6256895, -1.2473136, -2.1598206, 2.1596746
8: -0.1769443, 0.7755142, -0.1769488, 0.7758844, -0.8949457, 0.8945947
9: -1.5095581, 0.1987185, -1.5112215, 0.1977375, -1.1926892, 1.1891683

Time for backsubstitution: 4.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776727, upper bound: 0.5779288
time: 9.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776728, upper bound: 0.5779764
time: 10.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.9581757, 0.7207874, -0.9581684, 0.7207649, -1.4296409, 1.4296975
1: -1.0988555, 0.6274045, -1.0989852, 0.6274854, -0.8840921, 0.8850889
2: -3.6657467, -1.6741768, -3.6657467, -1.6741550, -1.3864824, 1.3864449
3: -4.1972675, -0.7895185, -4.1972752, -0.7895079, -1.5553920, 1.5555346
4: -5.0266848, -1.6267132, -5.0267253, -1.6267219, -2.2391853, 2.2377474
5: -4.3326907, -1.0016932, -4.3326879, -1.0017006, -1.5121548, 1.5118594
6: -8.4141445, -4.5934982, -8.4141693, -4.5935154, -1.5204643, 1.5211813
7: -4.6257486, -1.2472382, -4.6257892, -1.2472380, -2.1598549, 2.1597605
8: -0.1770551, 0.7755083, -0.1770646, 0.7755044, -0.8947253, 0.8946967
9: -1.5123194, 0.1982417, -1.5123826, 0.1983160, -1.1924734, 1.1915950

Time for backsubstitution: 4.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779615, upper bound: 0.5779358
time: 8.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779598, upper bound: 0.5779802
time: 9.30 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.9579984, 0.7207875, -0.9580150, 0.7209815, -1.4297874, 1.4296906
1: -1.0988536, 0.6265277, -1.1008788, 0.6263839, -0.8839569, 0.8883647
2: -3.6657028, -1.6741750, -3.6657190, -1.6739137, -1.3866135, 1.3864326
3: -4.1972642, -0.7896990, -4.1977248, -0.7897332, -1.5554117, 1.5557985
4: -5.0265431, -1.6267154, -5.0265918, -1.6261859, -2.2392397, 2.2377143
5: -4.3326616, -1.0019059, -4.3333678, -1.0019611, -1.5120873, 1.5125046
6: -8.4141388, -4.5938692, -8.4147453, -4.5939837, -1.5203998, 1.5214255
7: -4.6255331, -1.2472410, -4.6257620, -1.2472401, -2.1596556, 2.1597557
8: -0.1769859, 0.7755020, -0.1770040, 0.7759018, -0.8950994, 0.8946102
9: -1.5123150, 0.1977863, -1.5133922, 0.1977436, -1.1925373, 1.1923844

Time for backsubstitution: 4.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779818, upper bound: 0.5779324
time: 18.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779822, upper bound: 0.5779793
time: 14.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 38.20 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5775897, upper bound: 0.5773421
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5775878, upper bound: 0.5773854
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776083, upper bound: 0.5773426
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776088, upper bound: 0.5773875
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5778975, upper bound: 0.5773463
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5778980, upper bound: 0.5773942
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779183, upper bound: 0.5773465
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779175, upper bound: 0.5773957
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5775870, upper bound: 0.5777592
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5775872, upper bound: 0.5778071
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776060, upper bound: 0.5777576
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776089, upper bound: 0.5778032
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5778969, upper bound: 0.5777654
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5778983, upper bound: 0.5778137
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779156, upper bound: 0.5777658
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779179, upper bound: 0.5778106
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776538, upper bound: 0.5775105
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776511, upper bound: 0.5775586
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776724, upper bound: 0.5775113
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776721, upper bound: 0.5775571
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779622, upper bound: 0.5775142
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779622, upper bound: 0.5775599
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779819, upper bound: 0.5775134
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779827, upper bound: 0.5775628
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776525, upper bound: 0.5779264
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776525, upper bound: 0.5779700
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776727, upper bound: 0.5779288
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5776728, upper bound: 0.5779764
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779615, upper bound: 0.5779358
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779598, upper bound: 0.5779802
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779818, upper bound: 0.5779324
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.20
Output dim: 1, lower bound: -0.5779822, upper bound: 0.5779793

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.9569726, 0.7196602, -0.9566587, 0.7198637, -1.4278944, 1.4277308
1: -1.0920551, 0.6255831, -1.0913411, 0.6222293, -0.8756081, 0.8786662
2: -3.6629100, -1.6739522, -3.6632924, -1.6741942, -1.3828163, 1.3836480
3: -4.1935964, -0.7899058, -4.1941347, -0.7901292, -1.5496325, 1.5504053
4: -5.0193138, -1.6290429, -5.0202303, -1.6292353, -2.2277603, 2.2294755
5: -4.3291988, -1.0024031, -4.3297639, -1.0025369, -1.5058326, 1.5066583
6: -8.4106846, -4.5941319, -8.4104357, -4.5953650, -1.5156524, 1.5167782
7: -4.6204796, -1.2473406, -4.6203861, -1.2473896, -2.1515570, 2.1524830
8: -0.1766447, 0.7753282, -0.1766891, 0.7752463, -0.8939355, 0.8940748
9: -1.5090001, 0.1968884, -1.5084136, 0.1952115, -1.1884937, 1.1884527

Time for backsubstitution: 4.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5774979, upper bound: 0.5768972
time: 292.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5774976, upper bound: 0.5772523
time: 39.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.9578131, 0.7199363, -0.9570127, 0.7198734, -1.4287348, 1.4286543
1: -1.0982628, 0.6259143, -1.0914074, 0.6224687, -0.8836537, 0.8787754
2: -3.6646233, -1.6696631, -3.6643009, -1.6741942, -1.3844621, 1.3906233
3: -4.1932034, -0.7872132, -4.1935701, -0.7901210, -1.5507348, 1.5628384
4: -5.0238142, -1.6195238, -5.0231233, -1.6292341, -2.2299209, 2.2488894
5: -4.3287702, -1.0000914, -4.3291907, -1.0025344, -1.5066571, 1.5186089
6: -8.4102430, -4.5928845, -8.4100571, -4.5953560, -1.5156925, 1.5231264
7: -4.6230960, -1.2453613, -4.6218019, -1.2473918, -2.1554892, 2.1632757
8: -0.1770695, 0.7755491, -0.1766855, 0.7753870, -0.8945698, 0.8942109
9: -1.5153928, 0.1969167, -1.5085239, 0.1952154, -1.1967971, 1.1885083

Time for backsubstitution: 4.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5774977, upper bound: 0.5769483
time: 37.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5774986, upper bound: 0.5773013
time: 25.28 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.9567951, 0.7196607, -0.9565053, 0.7200801, -1.4280413, 1.4277233
1: -1.0920521, 0.6247061, -1.0932345, 0.6211276, -0.8754714, 0.8819419
2: -3.6628652, -1.6739513, -3.6632643, -1.6739533, -1.3829480, 1.3836358
3: -4.1935935, -0.7900852, -4.1945848, -0.7903523, -1.5496526, 1.5506703
4: -5.0191727, -1.6290467, -5.0200977, -1.6286998, -2.2278132, 2.2294421
5: -4.3291702, -1.0026176, -4.3304443, -1.0027965, -1.5057664, 1.5073031
6: -8.4106750, -4.5945020, -8.4110117, -4.5958328, -1.5155873, 1.5170228
7: -4.6202641, -1.2473423, -4.6203589, -1.2473922, -2.1513574, 2.1524792
8: -0.1765754, 0.7753220, -0.1766287, 0.7756438, -0.8943096, 0.8939884
9: -1.5089953, 0.1964333, -1.5094242, 0.1946394, -1.1885570, 1.1892421

Time for backsubstitution: 4.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775191, upper bound: 0.5769005
time: 33.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775200, upper bound: 0.5772557
time: 8.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.9576356, 0.7199366, -0.9568594, 0.7200896, -1.4288808, 1.4286473
1: -1.0982599, 0.6250374, -1.0933005, 0.6213675, -0.8835167, 0.8820511
2: -3.6645796, -1.6696620, -3.6642728, -1.6739535, -1.3845937, 1.3906108
3: -4.1932011, -0.7873929, -4.1940203, -0.7903447, -1.5507548, 1.5631030
4: -5.0236726, -1.6195270, -5.0229902, -1.6286985, -2.2299745, 2.2488558
5: -4.3287430, -1.0003048, -4.3298712, -1.0027950, -1.5065900, 1.5192530
6: -8.4102364, -4.5932579, -8.4106312, -4.5958228, -1.5156279, 1.5233703
7: -4.6228809, -1.2453632, -4.6217742, -1.2473936, -2.1552896, 2.1632717
8: -0.1770002, 0.7755427, -0.1766251, 0.7757844, -0.8949437, 0.8941244
9: -1.5153877, 0.1964613, -1.5095341, 0.1946433, -1.1968602, 1.1892977

Time for backsubstitution: 4.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775183, upper bound: 0.5769458
time: 18.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775194, upper bound: 0.5772997
time: 23.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.9566391, 0.7206773, -0.9567095, 0.7205669, -1.4285400, 1.4280262
1: -1.0987864, 0.6234454, -1.0963024, 0.6222405, -0.8740599, 0.8857200
2: -3.6626682, -1.6739609, -3.6633120, -1.6741831, -1.3828120, 1.3837879
3: -4.1940370, -0.7900946, -4.1944785, -0.7901204, -1.5494606, 1.5509646
4: -5.0190382, -1.6260945, -5.0203013, -1.6272016, -2.2302790, 2.2295532
5: -4.3294721, -1.0026352, -4.3299980, -1.0025162, -1.5056046, 1.5071610
6: -8.4127712, -4.5946608, -8.4119873, -4.5953522, -1.5153372, 1.5186135
7: -4.6196423, -1.2472173, -4.6204567, -1.2472947, -2.1513526, 2.1525173
8: -0.1766865, 0.7753170, -0.1767450, 0.7752646, -0.8940903, 0.8940917
9: -1.5117706, 0.1959563, -1.5106037, 0.1952178, -1.1876644, 1.1910405

Time for backsubstitution: 4.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778082, upper bound: 0.5769084
time: 97.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778078, upper bound: 0.5772558
time: 8.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.9574795, 0.7209531, -0.9570638, 0.7205763, -1.4293797, 1.4289504
1: -1.1049933, 0.6237767, -1.0963686, 0.6224799, -0.8821044, 0.8858289
2: -3.6643832, -1.6696720, -3.6643200, -1.6741831, -1.3844573, 1.3907632
3: -4.1936440, -0.7874012, -4.1939130, -0.7901115, -1.5505624, 1.5633979
4: -5.0235391, -1.6165748, -5.0231953, -1.6272023, -2.2324402, 2.2489660
5: -4.3290453, -1.0003219, -4.3294253, -1.0025148, -1.5064282, 1.5191119
6: -8.4123306, -4.5934124, -8.4116068, -4.5953436, -1.5153778, 1.5249615
7: -4.6222591, -1.2452393, -4.6218739, -1.2472969, -2.1552844, 2.1633098
8: -0.1771114, 0.7755378, -0.1767413, 0.7754053, -0.8947247, 0.8942277
9: -1.5181608, 0.1959845, -1.5107138, 0.1952218, -1.1959671, 1.1910964

Time for backsubstitution: 4.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778088, upper bound: 0.5769505
time: 17.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778084, upper bound: 0.5773037
time: 49.15 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.9564615, 0.7206778, -0.9565560, 0.7207831, -1.4286865, 1.4280190
1: -1.0987840, 0.6225688, -1.0981958, 0.6211391, -0.8739231, 0.8889955
2: -3.6626241, -1.6739593, -3.6632841, -1.6739422, -1.3829434, 1.3837752
3: -4.1940336, -0.7902741, -4.1949296, -0.7903433, -1.5494806, 1.5512292
4: -5.0188971, -1.6260976, -5.0201697, -1.6266682, -2.2303326, 2.2295191
5: -4.3294439, -1.0028485, -4.3306785, -1.0027769, -1.5055377, 1.5078058
6: -8.4127607, -4.5950336, -8.4125633, -4.5958204, -1.5152724, 1.5188572
7: -4.6194272, -1.2472196, -4.6204300, -1.2472966, -2.1511531, 2.1525128
8: -0.1766172, 0.7753108, -0.1766846, 0.7756621, -0.8944644, 0.8940048
9: -1.5117660, 0.1955012, -1.5116129, 0.1946458, -1.1877275, 1.1918294

Time for backsubstitution: 4.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778279, upper bound: 0.5769042
time: 12.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778280, upper bound: 0.5772585
time: 10.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.9573021, 0.7209534, -0.9569103, 0.7207928, -1.4295260, 1.4289432
1: -1.1049907, 0.6229002, -1.0982621, 0.6213789, -0.8819678, 0.8891049
2: -3.6643384, -1.6696711, -3.6642923, -1.6739422, -1.3845887, 1.3907511
3: -4.1936412, -0.7875814, -4.1943641, -0.7903345, -1.5505822, 1.5636621
4: -5.0233974, -1.6165789, -5.0230622, -1.6266674, -2.2324934, 2.2489319
5: -4.3290167, -1.0005356, -4.3301058, -1.0027742, -1.5063611, 1.5197564
6: -8.4123230, -4.5937853, -8.4121838, -4.5958109, -1.5153122, 1.5252056
7: -4.6220446, -1.2452413, -4.6218472, -1.2472992, -2.1550851, 2.1633058
8: -0.1770424, 0.7755316, -0.1766809, 0.7758029, -0.8950987, 0.8941408
9: -1.5181559, 0.1955294, -1.5117241, 0.1946498, -1.1960301, 1.1918857

Time for backsubstitution: 4.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778272, upper bound: 0.5769516
time: 15.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778283, upper bound: 0.5773074
time: 14.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.9569812, 0.7198664, -0.9569095, 0.7201299, -1.4278200, 1.4280787
1: -1.0938268, 0.6255850, -1.0937326, 0.6244347, -0.8789467, 0.8763815
2: -3.6629143, -1.6739476, -3.6633070, -1.6741784, -1.3828661, 1.3836530
3: -4.1938224, -0.7899047, -4.1944313, -0.7899570, -1.5498276, 1.5503632
4: -5.0193243, -1.6284090, -5.0206609, -1.6285890, -2.2272925, 2.2301164
5: -4.3292875, -1.0024014, -4.3299232, -1.0024357, -1.5061297, 1.5064459
6: -8.4115820, -4.5941277, -8.4115887, -4.5944295, -1.5160832, 1.5164644
7: -4.6204925, -1.2472916, -4.6208820, -1.2473283, -2.1515079, 2.1530285
8: -0.1766568, 0.7753311, -0.1767138, 0.7752706, -0.8939744, 0.8941108
9: -1.5098946, 0.1968892, -1.5096980, 0.1965196, -1.1901449, 1.1873649

Time for backsubstitution: 4.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5774982, upper bound: 0.5773197
time: 11.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5774994, upper bound: 0.5776701
time: 37.46 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.9578221, 0.7201424, -0.9572638, 0.7201395, -1.4286598, 1.4290020
1: -1.1000352, 0.6259160, -1.0937991, 0.6246742, -0.8869843, 0.8764912
2: -3.6646276, -1.6696589, -3.6643150, -1.6741784, -1.3845122, 1.3906282
3: -4.1934299, -0.7872132, -4.1938667, -0.7899488, -1.5509298, 1.5627966
4: -5.0238237, -1.6188904, -5.0235538, -1.6285882, -2.2294545, 2.2495294
5: -4.3288593, -1.0000890, -4.3293514, -1.0024344, -1.5069542, 1.5183967
6: -8.4111423, -4.5928822, -8.4112101, -4.5944204, -1.5161231, 1.5228120
7: -4.6231093, -1.2453133, -4.6222982, -1.2473302, -2.1554403, 2.1638222
8: -0.1770819, 0.7755522, -0.1767100, 0.7754114, -0.8946085, 0.8942472
9: -1.5162861, 0.1969174, -1.5098088, 0.1965239, -1.1984421, 1.1874213

Time for backsubstitution: 4.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5774978, upper bound: 0.5773659
time: 38.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5774977, upper bound: 0.5777167
time: 16.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.9568040, 0.7198668, -0.9567563, 0.7203462, -1.4279667, 1.4280713
1: -1.0938246, 0.6247084, -1.0956259, 0.6233337, -0.8788103, 0.8796564
2: -3.6628702, -1.6739466, -3.6632788, -1.6739374, -1.3829978, 1.3836404
3: -4.1938186, -0.7900848, -4.1948833, -0.7901810, -1.5498472, 1.5506279
4: -5.0191817, -1.6284127, -5.0205278, -1.6280539, -2.2273469, 2.2300832
5: -4.3292575, -1.0026164, -4.3306046, -1.0026964, -1.5060630, 1.5070904
6: -8.4115734, -4.5945005, -8.4121656, -4.5948973, -1.5160183, 1.5167090
7: -4.6202774, -1.2472944, -4.6208549, -1.2473297, -2.1513081, 2.1530242
8: -0.1765876, 0.7753249, -0.1766533, 0.7756681, -0.8943484, 0.8940242
9: -1.5098901, 0.1964339, -1.5107077, 0.1959479, -1.1902087, 1.1881536

Time for backsubstitution: 4.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775183, upper bound: 0.5773148
time: 85.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775178, upper bound: 0.5776667
time: 10.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.9576447, 0.7201425, -0.9571105, 0.7203557, -1.4288063, 1.4289955
1: -1.1000323, 0.6250391, -1.0956926, 0.6235733, -0.8868479, 0.8797665
2: -3.6645839, -1.6696581, -3.6642871, -1.6739377, -1.3846436, 1.3906157
3: -4.1934266, -0.7873919, -4.1943169, -0.7901726, -1.5509493, 1.5630610
4: -5.0236821, -1.6188927, -5.0234213, -1.6280532, -2.2295084, 2.2494965
5: -4.3288302, -1.0003029, -4.3300314, -1.0026944, -1.5068874, 1.5190411
6: -8.4111347, -4.5932550, -8.4117870, -4.5948873, -1.5160587, 1.5230566
7: -4.6228943, -1.2453156, -4.6222711, -1.2473325, -2.1552410, 2.1638179
8: -0.1770127, 0.7755458, -0.1766495, 0.7758088, -0.8949826, 0.8941603
9: -1.5162816, 0.1964621, -1.5108186, 0.1959518, -1.1985052, 1.1882102

Time for backsubstitution: 4.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775191, upper bound: 0.5773611
time: 20.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775190, upper bound: 0.5777152
time: 13.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.9566479, 0.7208139, -0.9569604, 0.7207436, -1.4284954, 1.4283731
1: -1.1004993, 0.6234473, -1.0986320, 0.6244459, -0.8787509, 0.8847604
2: -3.6626728, -1.6739566, -3.6633260, -1.6741673, -1.3828621, 1.3837922
3: -4.1942406, -0.7900945, -4.1947522, -0.7899481, -1.5498046, 1.5510221
4: -5.0190477, -1.6257746, -5.0207324, -1.6267461, -2.2299514, 2.2302003
5: -4.3295393, -1.0026330, -4.3301382, -1.0024161, -1.5060817, 1.5070770
6: -8.4135036, -4.5946569, -8.4129629, -4.5944166, -1.5162609, 1.5188086
7: -4.6196556, -1.2471892, -4.6209526, -1.2472543, -2.1513424, 2.1531100
8: -0.1766982, 0.7753192, -0.1767689, 0.7752881, -0.8941281, 0.8941265
9: -1.5126513, 0.1959571, -1.5118686, 0.1965263, -1.1899920, 1.1905800

Time for backsubstitution: 4.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778086, upper bound: 0.5773241
time: 14.22 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778082, upper bound: 0.5776786
time: 29.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.9574881, 0.7210898, -0.9573147, 0.7207533, -1.4293356, 1.4292967
1: -1.1067052, 0.6237785, -1.0986986, 0.6246858, -0.8867889, 0.8848697
2: -3.6643867, -1.6696684, -3.6643345, -1.6741679, -1.3845077, 1.3907678
3: -4.1938496, -0.7874022, -4.1941881, -0.7899392, -1.5509062, 1.5634552
4: -5.0235486, -1.6162555, -5.0236249, -1.6267447, -2.2321126, 2.2496138
5: -4.3291116, -1.0003203, -4.3295660, -1.0024143, -1.5069057, 1.5190279
6: -8.4130630, -4.5934100, -8.4125843, -4.5944090, -1.5163010, 1.5251570
7: -4.6222720, -1.2452114, -4.6223698, -1.2472564, -2.1552749, 2.1639030
8: -0.1771235, 0.7755401, -0.1767653, 0.7754288, -0.8947624, 0.8942626
9: -1.5190399, 0.1959853, -1.5119792, 0.1965305, -1.1982884, 1.1906364

Time for backsubstitution: 4.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778075, upper bound: 0.5773715
time: 34.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778088, upper bound: 0.5777246
time: 19.24 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.9564700, 0.7208143, -0.9568073, 0.7209601, -1.4286420, 1.4283655
1: -1.1004965, 0.6225706, -1.1005259, 0.6233447, -0.8786143, 0.8880363
2: -3.6626289, -1.6739559, -3.6632984, -1.6739261, -1.3829937, 1.3837804
3: -4.1942382, -0.7902737, -4.1952033, -0.7901710, -1.5498245, 1.5512872
4: -5.0189061, -1.6257787, -5.0205989, -1.6262115, -2.2300050, 2.2301672
5: -4.3295093, -1.0028471, -4.3308201, -1.0026758, -1.5060154, 1.5077220
6: -8.4134922, -4.5950294, -8.4135389, -4.5948849, -1.5161963, 1.5190530
7: -4.6194401, -1.2471907, -4.6209259, -1.2472566, -2.1511433, 2.1531067
8: -0.1766293, 0.7753128, -0.1767085, 0.7756857, -0.8945020, 0.8940398
9: -1.5126472, 0.1955017, -1.5128782, 0.1959540, -1.1900554, 1.1913692

Time for backsubstitution: 4.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778271, upper bound: 0.5773248
time: 10.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778285, upper bound: 0.5776775
time: 9.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.9573104, 0.7210901, -0.9571610, 0.7209696, -1.4294819, 1.4292896
1: -1.1067021, 0.6229018, -1.1005924, 0.6235844, -0.8866517, 0.8881457
2: -3.6643422, -1.6696670, -3.6643064, -1.6739265, -1.3846391, 1.3907559
3: -4.1938467, -0.7875817, -4.1946378, -0.7901622, -1.5509261, 1.5637197
4: -5.0234065, -1.6162597, -5.0234923, -1.6262097, -2.2321663, 2.2495799
5: -4.3290825, -1.0005338, -4.3302460, -1.0026739, -1.5068384, 1.5196722
6: -8.4130554, -4.5937829, -8.4131603, -4.5948753, -1.5162361, 1.5254015
7: -4.6220565, -1.2452135, -4.6223421, -1.2472581, -2.1550756, 2.1638989
8: -0.1770542, 0.7755337, -0.1767049, 0.7758263, -0.8951362, 0.8941756
9: -1.5190350, 0.1955298, -1.5129888, 0.1959583, -1.1983513, 1.1914254

Time for backsubstitution: 4.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778282, upper bound: 0.5773723
time: 15.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778286, upper bound: 0.5777263
time: 19.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.9580010, 0.7196137, -0.9574844, 0.7198690, -1.4284836, 1.4286110
1: -1.0901638, 0.6291704, -1.0914987, 0.6249781, -0.8805317, 0.8784964
2: -3.6631074, -1.6741730, -3.6634624, -1.6741827, -1.3830551, 1.3837039
3: -4.1933289, -0.7893426, -4.1941452, -0.7897005, -1.5498495, 1.5507023
4: -5.0207815, -1.6299920, -5.0213933, -1.6292193, -2.2284632, 2.2303681
5: -4.3290610, -1.0014679, -4.3298044, -1.0018246, -1.5064180, 1.5071628
6: -8.4097528, -4.5930080, -8.4104443, -4.5944939, -1.5169797, 1.5169367
7: -4.6205463, -1.2473996, -4.6204872, -1.2473820, -2.1518326, 2.1526973
8: -0.1769757, 0.7752808, -0.1769646, 0.7752761, -0.8943086, 0.8943753
9: -1.5082448, 0.1991479, -1.5085946, 0.1969820, -1.1907657, 1.1892893

Time for backsubstitution: 4.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775635, upper bound: 0.5770658
time: 20.29 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775637, upper bound: 0.5774193
time: 67.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.9588413, 0.7198895, -0.9578386, 0.7198783, -1.4293234, 1.4295344
1: -1.0963734, 0.6295018, -1.0915658, 0.6252182, -0.8885608, 0.8786058
2: -3.6648226, -1.6698840, -3.6644709, -1.6741831, -1.3847013, 1.3906798
3: -4.1929374, -0.7866498, -4.1935811, -0.7896913, -1.5509515, 1.5631351
4: -5.0252824, -1.6204729, -5.0242867, -1.6292175, -2.2306242, 2.2497811
5: -4.3286343, -0.9991550, -4.3292322, -1.0018220, -1.5072420, 1.5191140
6: -8.4093132, -4.5917625, -8.4100637, -4.5944853, -1.5170197, 1.5232849
7: -4.6231627, -1.2454209, -4.6219034, -1.2473838, -2.1557662, 2.1634908
8: -0.1774008, 0.7755017, -0.1769611, 0.7754168, -0.8949429, 0.8945116
9: -1.5146402, 0.1991762, -1.5087049, 0.1969856, -1.1990608, 1.1893451

Time for backsubstitution: 4.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775644, upper bound: 0.5771123
time: 107.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775638, upper bound: 0.5774685
time: 49.18 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.9578237, 0.7196140, -0.9573313, 0.7200851, -1.4286301, 1.4286034
1: -1.0901613, 0.6282938, -1.0933925, 0.6238768, -0.8803960, 0.8817720
2: -3.6630647, -1.6741722, -3.6634347, -1.6739411, -1.3831868, 1.3836917
3: -4.1933274, -0.7895222, -4.1945977, -0.7899235, -1.5498691, 1.5509667
4: -5.0206413, -1.6299942, -5.0212593, -1.6286838, -2.2285166, 2.2303348
5: -4.3290329, -1.0016809, -4.3304853, -1.0020843, -1.5063505, 1.5078073
6: -8.4097443, -4.5933800, -8.4110193, -4.5949616, -1.5169148, 1.5171812
7: -4.6203308, -1.2474014, -4.6204610, -1.2473840, -2.1516328, 2.1526933
8: -0.1769065, 0.7752745, -0.1769041, 0.7756735, -0.8946830, 0.8942888
9: -1.5082400, 0.1986926, -1.5096042, 0.1964096, -1.1908289, 1.1900791

Time for backsubstitution: 4.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775828, upper bound: 0.5770684
time: 36.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775832, upper bound: 0.5774213
time: 39.00 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.9586635, 0.7198899, -0.9576851, 0.7200949, -1.4294690, 1.4295278
1: -1.0963702, 0.6286252, -1.0934584, 0.6241165, -0.8884249, 0.8818806
2: -3.6647785, -1.6698830, -3.6644423, -1.6739414, -1.3848323, 1.3906671
3: -4.1929336, -0.7868305, -4.1940308, -0.7899154, -1.5509712, 1.5633996
4: -5.0251408, -1.6204755, -5.0241528, -1.6286831, -2.2306769, 2.2497482
5: -4.3286057, -0.9993677, -4.3299131, -1.0020819, -1.5071746, 1.5197587
6: -8.4093065, -4.5921359, -8.4106407, -4.5949526, -1.5169547, 1.5235291
7: -4.6229477, -1.2454233, -4.6218758, -1.2473855, -2.1555660, 2.1634867
8: -0.1773315, 0.7754953, -0.1769005, 0.7758143, -0.8953172, 0.8944250
9: -1.5146348, 0.1987206, -1.5097150, 0.1964138, -1.1991233, 1.1901346

Time for backsubstitution: 4.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2050

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775842, upper bound: 0.5771133
time: 587.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5775833, upper bound: 0.5774646
time: 48.61 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 641.30 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5774979, upper bound: 0.5768972
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5774976, upper bound: 0.5772523
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5774977, upper bound: 0.5769483
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5774986, upper bound: 0.5773013
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775191, upper bound: 0.5769005
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775200, upper bound: 0.5772557
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775183, upper bound: 0.5769458
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775194, upper bound: 0.5772997
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778082, upper bound: 0.5769084
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778078, upper bound: 0.5772558
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778088, upper bound: 0.5769505
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778084, upper bound: 0.5773037
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778279, upper bound: 0.5769042
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778280, upper bound: 0.5772585
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778272, upper bound: 0.5769516
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778283, upper bound: 0.5773074
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5774982, upper bound: 0.5773197
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5774994, upper bound: 0.5776701
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5774978, upper bound: 0.5773659
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5774977, upper bound: 0.5777167
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775183, upper bound: 0.5773148
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775178, upper bound: 0.5776667
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775191, upper bound: 0.5773611
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775190, upper bound: 0.5777152
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778086, upper bound: 0.5773241
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778082, upper bound: 0.5776786
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778075, upper bound: 0.5773715
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778088, upper bound: 0.5777246
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778271, upper bound: 0.5773248
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778285, upper bound: 0.5776775
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778282, upper bound: 0.5773723
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5778286, upper bound: 0.5777263
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775635, upper bound: 0.5770658
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775637, upper bound: 0.5774193
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775644, upper bound: 0.5771123
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775638, upper bound: 0.5774685
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775828, upper bound: 0.5770684
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775832, upper bound: 0.5774213
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775842, upper bound: 0.5771133
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 641.30
Output dim: 1, lower bound: -0.5775833, upper bound: 0.5774646
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5779622, upper bound: 0.5775142
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5779622, upper bound: 0.5775599
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5779819, upper bound: 0.5775134
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5779827, upper bound: 0.5775628
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5776525, upper bound: 0.5779264
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5776525, upper bound: 0.5779700
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5776727, upper bound: 0.5779288
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5776728, upper bound: 0.5779764
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5779615, upper bound: 0.5779358
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5779598, upper bound: 0.5779802
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5779818, upper bound: 0.5779324
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 641.30
Output dim: 1, lower bound: -0.5779822, upper bound: 0.5779793
Binary search (step 0): status=Status.UNKNOWN, k_low=2, k_high=8, k_mid=5, eps_mid=0.0195312, abs_max=0.8949273824691772
rel_dist={1: [-0.579017517234431, 0.5790172832286323]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2522

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232011, upper bound: 0.5231409
time: 42.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232236, upper bound: 0.5232258
time: 42.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 85.07 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 85.07
Output dim: 1, lower bound: -0.5232011, upper bound: 0.5231409
IS_A2, status: Status.UNKNOWN, split count: 1, time: 85.07
Output dim: 1, lower bound: -0.5232236, upper bound: 0.5232258

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.9573896, 0.7210121, -0.9575496, 0.7209604, -1.4171439, 1.4174263
1: -1.1015255, 0.6246668, -1.0994729, 0.6254504, -0.8590677, 0.8589557
2: -3.6656461, -1.6738957, -3.6656780, -1.6741288, -1.3589841, 1.3591809
3: -4.1976347, -0.7899210, -4.1973557, -0.7897971, -1.4946210, 1.4942160
4: -5.0255575, -1.6253605, -5.0258322, -1.6263250, -2.1942601, 2.1949983
5: -4.3330030, -1.0024294, -4.3328266, -1.0022483, -1.4501176, 1.4496938
6: -8.4153233, -4.5942554, -8.4143839, -4.5940218, -1.4430964, 1.4427447
7: -4.6257863, -1.2470965, -4.6257896, -1.2471650, -2.1156907, 2.1157684
8: -0.1768111, 0.7755800, -0.1768552, 0.7755020, -0.8895242, 0.8896207
9: -1.5134718, 0.1961916, -1.5125322, 0.1965841, -1.1686883, 1.1677581

Time for backsubstitution: 4.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2080

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231661, upper bound: 0.5227732
time: 11.73 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231658, upper bound: 0.5231053
time: 42.11 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.9584174, 0.7209653, -0.9584627, 0.7209659, -1.4177289, 1.4184194
1: -1.0996343, 0.6282543, -1.0996485, 0.6284931, -0.8644072, 0.8587732
2: -3.6658449, -1.6741171, -3.6658635, -1.6741166, -1.3592272, 1.3592579
3: -4.1973686, -0.7893600, -4.1973691, -0.7893192, -1.4948895, 1.4944997
4: -5.0270252, -1.6263089, -5.0271235, -1.6263074, -2.1949563, 2.1960392
5: -4.3328667, -1.0014926, -4.3328724, -1.0014549, -1.4507849, 1.4501864
6: -8.4143925, -4.5931339, -8.4143934, -4.5930648, -1.4445579, 1.4428926
7: -4.6258521, -1.2471557, -4.6258955, -1.2471547, -2.1159697, 2.1160069
8: -0.1771424, 0.7755325, -0.1771611, 0.7755349, -0.8899010, 0.8899552
9: -1.5127169, 0.1984509, -1.5127338, 0.1985432, -1.1712540, 1.1685917

Time for backsubstitution: 4.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2080

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231844, upper bound: 0.5228606
time: 47.72 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231885, upper bound: 0.5231922
time: 33.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 85.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 85.45
Output dim: 1, lower bound: -0.5231661, upper bound: 0.5227732
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 85.45
Output dim: 1, lower bound: -0.5231658, upper bound: 0.5231053
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 85.45
Output dim: 1, lower bound: -0.5231844, upper bound: 0.5228606
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 85.45
Output dim: 1, lower bound: -0.5231885, upper bound: 0.5231922

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.9573650, 0.7207100, -0.9572811, 0.7206079, -1.4167776, 1.4169089
1: -1.0992403, 0.6246604, -1.0967007, 0.6232406, -0.8559456, 0.8569982
2: -3.6656373, -1.6739056, -3.6656594, -1.6741509, -1.3589246, 1.3591478
3: -4.1973457, -0.7899246, -4.1970158, -0.7899704, -1.4943863, 1.4939280
4: -5.0255308, -1.6260840, -5.0253839, -1.6271569, -2.1935420, 2.1940441
5: -4.3328071, -1.0024357, -4.3325539, -1.0023538, -1.4497032, 1.4493630
6: -8.4143229, -4.5942616, -8.4132252, -4.5949602, -1.4425113, 1.4420211
7: -4.6257582, -1.2471836, -4.6252785, -1.2472689, -2.1155629, 2.1151936
8: -0.1767785, 0.7755729, -0.1768088, 0.7754730, -0.8894392, 0.8895407
9: -1.5122921, 0.1961892, -1.5110326, 0.1952735, -1.1669998, 1.1665804

Time for backsubstitution: 4.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5228626, upper bound: 0.5227411
time: 44.99 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5230521, upper bound: 0.5227431
time: 41.46 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.9573744, 0.7208612, -0.9575323, 0.7207847, -1.4167196, 1.4172697
1: -1.1011554, 0.6246627, -1.0990421, 0.6254458, -0.8582799, 0.8559237
2: -3.6656418, -1.6739016, -3.6656735, -1.6741352, -1.3589743, 1.3591516
3: -4.1975846, -0.7899240, -4.1972976, -0.7897978, -1.4945115, 1.4939724
4: -5.0255418, -1.6256428, -5.0258141, -1.6266532, -2.1932220, 2.1946225
5: -4.3328934, -1.0024337, -4.3326988, -1.0022535, -1.4500015, 1.4492729
6: -8.4151897, -4.5942574, -8.4142294, -4.5940247, -1.4425209, 1.4421601
7: -4.6257730, -1.2471522, -4.6257739, -1.2472286, -2.1155512, 2.1156993
8: -0.1767918, 0.7755754, -0.1768329, 0.7754965, -0.8894778, 0.8895739
9: -1.5132744, 0.1961897, -1.5123029, 0.1965818, -1.1682992, 1.1660582

Time for backsubstitution: 4.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5228661, upper bound: 0.5229869
time: 159.88 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5230473, upper bound: 0.5229916
time: 24.61 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.9583930, 0.7206630, -0.9581940, 0.7206131, -1.4173625, 1.4179020
1: -1.0973495, 0.6282479, -1.0968773, 0.6262829, -0.8612854, 0.8568164
2: -3.6658359, -1.6741267, -3.6658459, -1.6741383, -1.3591678, 1.3592240
3: -4.1970778, -0.7893633, -4.1970286, -0.7894936, -1.4946548, 1.4942114
4: -5.0269990, -1.6270329, -5.0266733, -1.6271393, -2.1942387, 2.1950841
5: -4.3326716, -1.0015001, -4.3325977, -1.0015612, -1.4503702, 1.4498556
6: -8.4133921, -4.5931396, -8.4132338, -4.5940027, -1.4439728, 1.4421680
7: -4.6258249, -1.2472427, -4.6253843, -1.2472608, -2.1158421, 2.1154323
8: -0.1771097, 0.7755254, -0.1771146, 0.7755060, -0.8898159, 0.8898753
9: -1.5115376, 0.1984488, -1.5112342, 0.1972324, -1.1695656, 1.1674153

Time for backsubstitution: 4.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5228893, upper bound: 0.5228255
time: 13.66 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5230752, upper bound: 0.5228267
time: 41.81 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.9584021, 0.7208143, -0.9584451, 0.7207901, -1.4173046, 1.4182627
1: -1.0992644, 0.6282500, -1.0992179, 0.6284883, -0.8636193, 0.8557419
2: -3.6658411, -1.6741228, -3.6658595, -1.6741222, -1.3592174, 1.3592286
3: -4.1973186, -0.7893611, -4.1973100, -0.7893208, -1.4947805, 1.4942555
4: -5.0270090, -1.6265913, -5.0271049, -1.6266359, -2.1939178, 2.1956625
5: -4.3327570, -1.0014973, -4.3327451, -1.0014597, -1.4506681, 1.4497658
6: -8.4142580, -4.5931344, -8.4142389, -4.5930676, -1.4439824, 1.4423074
7: -4.6258388, -1.2472110, -4.6258798, -1.2472196, -2.1158307, 2.1159384
8: -0.1771229, 0.7755278, -0.1771388, 0.7755294, -0.8898547, 0.8899084
9: -1.5125198, 0.1984493, -1.5125045, 0.1985409, -1.1708649, 1.1668926

Time for backsubstitution: 4.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5228896, upper bound: 0.5230718
time: 13.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5230722, upper bound: 0.5230754
time: 63.17 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 81.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 81.73
Output dim: 1, lower bound: -0.5228626, upper bound: 0.5227411
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 81.73
Output dim: 1, lower bound: -0.5230521, upper bound: 0.5227431
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 81.73
Output dim: 1, lower bound: -0.5228661, upper bound: 0.5229869
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 81.73
Output dim: 1, lower bound: -0.5230473, upper bound: 0.5229916
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 81.73
Output dim: 1, lower bound: -0.5228893, upper bound: 0.5228255
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 81.73
Output dim: 1, lower bound: -0.5230752, upper bound: 0.5228267
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 81.73
Output dim: 1, lower bound: -0.5228896, upper bound: 0.5230718
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 81.73
Output dim: 1, lower bound: -0.5230722, upper bound: 0.5230754

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.9576936, 0.7196521, -0.9572186, 0.7197754, -1.4159800, 1.4157357
1: -1.0920897, 0.6267965, -1.0907744, 0.6232266, -0.8465853, 0.8488194
2: -3.6658776, -1.6739464, -3.6656370, -1.6741922, -1.3588986, 1.3589635
3: -4.1968699, -0.7897371, -4.1966128, -0.7899832, -1.4936665, 1.4932572
4: -5.0258017, -1.6290909, -5.0252967, -1.6295354, -2.1906157, 2.1906629
5: -4.3324938, -1.0022066, -4.3322611, -1.0023786, -1.4491639, 1.4487627
6: -8.4121571, -4.5937328, -8.4114323, -4.5949769, -1.4395872, 1.4398522
7: -4.6265869, -1.2473379, -4.6251898, -1.2473998, -2.1157253, 2.1148496
8: -0.1767257, 0.7755836, -0.1767368, 0.7754514, -0.8893375, 0.8894947
9: -1.5093294, 0.1971201, -1.5084291, 0.1952655, -1.1637371, 1.1635935

Time for backsubstitution: 4.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5225967, upper bound: 0.5224872
time: 174.49 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5226134, upper bound: 0.5224858
time: 88.55 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.9573601, 0.7206728, -0.9572777, 0.7205818, -1.4167430, 1.4160275
1: -1.0988286, 0.6246590, -1.0964099, 0.6232396, -0.8448986, 0.8569779
2: -3.6656363, -1.6739550, -3.6656590, -1.6741855, -1.3589114, 1.3591082
3: -4.1973152, -0.7899261, -4.1969938, -0.7899711, -1.4934855, 1.4938958
4: -5.0255260, -1.6261328, -5.0253811, -1.6271917, -2.1934958, 2.1907001
5: -4.3327703, -1.0024385, -4.3325262, -1.0023547, -1.4489310, 1.4493356
6: -8.4142561, -4.5942636, -8.4131775, -4.5949616, -1.4392378, 1.4420148
7: -4.6257501, -1.2472141, -4.6252723, -1.2472916, -2.1155496, 2.1148934
8: -0.1767677, 0.7755724, -0.1768011, 0.7754726, -0.8894963, 0.8895172
9: -1.5121036, 0.1961880, -1.5108991, 0.1952727, -1.1628518, 1.1665239

Time for backsubstitution: 4.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5227808, upper bound: 0.5224879
time: 113.25 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5227994, upper bound: 0.5224871
time: 162.07 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.9577036, 0.7198798, -0.9574695, 0.7200413, -1.4158916, 1.4160988
1: -1.0940506, 0.6267987, -1.0931662, 0.6254321, -0.8499770, 0.8464187
2: -3.6658826, -1.6739414, -3.6656506, -1.6741766, -1.3589485, 1.3589675
3: -4.1971202, -0.7897365, -4.1969109, -0.7898105, -1.4938812, 1.4932010
4: -5.0258117, -1.6284057, -5.0257273, -1.6288490, -2.1901553, 2.1913304
5: -4.3325911, -1.0022049, -4.3324218, -1.0022787, -1.4494785, 1.4485435
6: -8.4131498, -4.5937300, -8.4125891, -4.5940409, -1.4400598, 1.4394825
7: -4.6266022, -1.2472844, -4.6256871, -1.2473335, -2.1156745, 2.1154032
8: -0.1767394, 0.7755870, -0.1767613, 0.7754761, -0.8893774, 0.8895289
9: -1.5103195, 0.1971211, -1.5097132, 0.1965738, -1.1654530, 1.1624438

Time for backsubstitution: 4.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5225976, upper bound: 0.5227362
time: 51.23 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5226132, upper bound: 0.5227372
time: 25.17 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.9573697, 0.7208238, -0.9575284, 0.7207583, -1.4166846, 1.4163888
1: -1.1007223, 0.6246613, -1.0987363, 0.6254451, -0.8496598, 0.8559028
2: -3.6656411, -1.6739507, -3.6656721, -1.6741700, -1.3589613, 1.3591118
3: -4.1975369, -0.7899250, -4.1972647, -0.7897995, -1.4938493, 1.4939398
4: -5.0255365, -1.6257812, -5.0258102, -1.6267508, -2.1931751, 2.1913657
5: -4.3328428, -1.0024351, -4.3326650, -1.0022548, -1.4494288, 1.4492452
6: -8.4150696, -4.5942588, -8.4141455, -4.5940266, -1.4401999, 1.4421538
7: -4.6257639, -1.2471825, -4.6257682, -1.2472502, -2.1155374, 2.1154943
8: -0.1767807, 0.7755749, -0.1768250, 0.7754961, -0.8895345, 0.8895507
9: -1.5130762, 0.1961888, -1.5121626, 0.1965814, -1.1652443, 1.1660014

Time for backsubstitution: 4.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5227799, upper bound: 0.5227399
time: 50.17 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5227998, upper bound: 0.5227367
time: 20.97 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.9587215, 0.7196054, -0.9581316, 0.7197807, -1.4165647, 1.4167289
1: -1.0901986, 0.6303842, -1.0909507, 0.6262689, -0.8519244, 0.8486373
2: -3.6660767, -1.6741678, -3.6658220, -1.6741791, -1.3591419, 1.3590400
3: -4.1966019, -0.7891743, -4.1966248, -0.7895045, -1.4939353, 1.4935402
4: -5.0272703, -1.6300399, -5.0265875, -1.6295173, -2.1913114, 2.1917024
5: -4.3323565, -1.0012703, -4.3323059, -1.0015855, -1.4498298, 1.4492552
6: -8.4112244, -4.5926123, -8.4114428, -4.5940199, -1.4410486, 1.4399996
7: -4.6266532, -1.2473972, -4.6252961, -1.2473905, -2.1160045, 2.1150887
8: -0.1770570, 0.7755362, -0.1770426, 0.7754846, -0.8897141, 0.8898290
9: -1.5085748, 0.1993799, -1.5086306, 0.1972243, -1.1663032, 1.1644282

Time for backsubstitution: 4.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5226206, upper bound: 0.5225731
time: 15.58 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5226366, upper bound: 0.5225714
time: 20.31 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.9583877, 0.7206259, -0.9581907, 0.7205871, -1.4173279, 1.4170210
1: -1.0969373, 0.6282468, -1.0965854, 0.6262820, -0.8502385, 0.8567953
2: -3.6658351, -1.6741760, -3.6658449, -1.6741722, -1.3591545, 1.3591845
3: -4.1970472, -0.7893636, -4.1970067, -0.7894939, -1.4937543, 1.4941795
4: -5.0269938, -1.6270803, -5.0266705, -1.6271740, -2.1941919, 2.1917400
5: -4.3326335, -1.0015010, -4.3325710, -1.0015622, -1.4495974, 1.4498283
6: -8.4133253, -4.5931406, -8.4131870, -4.5940037, -1.4406984, 1.4421620
7: -4.6258168, -1.2472732, -4.6253796, -1.2472817, -2.1158285, 2.1151326
8: -0.1770990, 0.7755250, -0.1771070, 0.7755057, -0.8898730, 0.8898520
9: -1.5113485, 0.1984479, -1.5111006, 0.1972318, -1.1654184, 1.1673576

Time for backsubstitution: 4.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5228063, upper bound: 0.5225764
time: 9.51 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5228234, upper bound: 0.5225751
time: 14.67 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.9587313, 0.7198334, -0.9583826, 0.7200468, -1.4164760, 1.4170918
1: -1.0921593, 0.6303862, -1.0933418, 0.6284745, -0.8553159, 0.8462373
2: -3.6660817, -1.6741629, -3.6658361, -1.6741633, -1.3591921, 1.3590443
3: -4.1968536, -0.7891734, -4.1969237, -0.7893329, -1.4941499, 1.4934845
4: -5.0272803, -1.6293532, -5.0270176, -1.6288307, -2.1908512, 2.1923704
5: -4.3324537, -1.0012679, -4.3324680, -1.0014851, -1.4501452, 1.4490359
6: -8.4122181, -4.5926075, -8.4125977, -4.5930848, -1.4415212, 1.4396300
7: -4.6266689, -1.2473435, -4.6257939, -1.2473239, -2.1159544, 2.1156425
8: -0.1770705, 0.7755395, -0.1770672, 0.7755091, -0.8897539, 0.8898635
9: -1.5095650, 0.1993805, -1.5099151, 0.1985330, -1.1680187, 1.1632783

Time for backsubstitution: 4.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5226239, upper bound: 0.5225735
time: 474.64 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5226391, upper bound: 0.5228217
time: 14.80 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.9583974, 0.7207770, -0.9584415, 0.7207637, -1.4172699, 1.4173825
1: -1.0988306, 0.6282489, -1.0989118, 0.6284875, -0.8549991, 0.8557208
2: -3.6658401, -1.6741714, -3.6658590, -1.6741567, -1.3592045, 1.3591886
3: -4.1972723, -0.7893624, -4.1972790, -0.7893215, -1.4941182, 1.4942231
4: -5.0270042, -1.6267288, -5.0271001, -1.6267328, -2.1938710, 2.1924062
5: -4.3327069, -1.0014998, -4.3327103, -1.0014608, -1.4500957, 1.4497378
6: -8.4141388, -4.5931368, -8.4141550, -4.5930696, -1.4416614, 1.4423010
7: -4.6258316, -1.2472425, -4.6258740, -1.2472414, -2.1158161, 2.1157329
8: -0.1771121, 0.7755273, -0.1771311, 0.7755291, -0.8899115, 0.8898852
9: -1.5123212, 0.1984484, -1.5123643, 0.1985403, -1.1678104, 1.1668351

Time for backsubstitution: 4.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5228046, upper bound: 0.5228268
time: 26.74 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5228230, upper bound: 0.5228249
time: 31.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 63.31 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5225967, upper bound: 0.5224872
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5226134, upper bound: 0.5224858
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5227808, upper bound: 0.5224879
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5227994, upper bound: 0.5224871
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5225976, upper bound: 0.5227362
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5226132, upper bound: 0.5227372
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5227799, upper bound: 0.5227399
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5227998, upper bound: 0.5227367
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5226206, upper bound: 0.5225731
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5226366, upper bound: 0.5225714
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5228063, upper bound: 0.5225764
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5228234, upper bound: 0.5225751
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5226239, upper bound: 0.5225735
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5226391, upper bound: 0.5228217
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5228046, upper bound: 0.5228268
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.31
Output dim: 1, lower bound: -0.5228230, upper bound: 0.5228249

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.9574405, 0.7196470, -0.9569165, 0.7197692, -1.4156773, 1.4153891
1: -1.0920776, 0.6258366, -1.0907600, 0.6220770, -0.8454204, 0.8478339
2: -3.6657724, -1.6739519, -3.6655126, -1.6741995, -1.3587797, 1.3588276
3: -4.1968589, -0.7899140, -4.1965985, -0.7901947, -1.4934989, 1.4931123
4: -5.0254369, -1.6291033, -5.0248609, -1.6295496, -2.1902938, 2.1902859
5: -4.3324652, -1.0024265, -4.3322258, -1.0026414, -1.4488616, 1.4484954
6: -8.4121494, -4.5941439, -8.4114246, -4.5954695, -1.4391965, 1.4395213
7: -4.6264925, -1.2473403, -4.6250772, -1.2474028, -2.1156154, 2.1147201
8: -0.1766590, 0.7755616, -0.1766568, 0.7754251, -0.8892224, 0.8893697
9: -1.5093056, 0.1968856, -1.5084003, 0.1949845, -1.1635060, 1.1633928

Time for backsubstitution: 4.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5225394, upper bound: 0.5223870
time: 13.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5225374, upper bound: 0.5224283
time: 213.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.9572473, 0.7196475, -0.9567631, 0.7199856, -1.4158206, 1.4153774
1: -1.0920745, 0.6248788, -1.0926530, 0.6209760, -0.8452699, 0.8510934
2: -3.6657252, -1.6739502, -3.6654840, -1.6739585, -1.3589058, 1.3588128
3: -4.1968565, -0.7901106, -4.1970510, -0.7904176, -1.4935179, 1.4933536
4: -5.0252862, -1.6291068, -5.0247278, -1.6290146, -2.1903214, 2.1902528
5: -4.3324347, -1.0026610, -4.3329067, -1.0029022, -1.4487828, 1.4491270
6: -8.4121408, -4.5945511, -8.4120007, -4.5959363, -1.4391232, 1.4397346
7: -4.6262541, -1.2473419, -4.6250505, -1.2474049, -2.1153955, 2.1147161
8: -0.1765834, 0.7755550, -0.1765966, 0.7758226, -0.8895898, 0.8892804
9: -1.5093007, 0.1963829, -1.5094110, 0.1944129, -1.1635456, 1.1641295

Time for backsubstitution: 4.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5225579, upper bound: 0.5223900
time: 156.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5225579, upper bound: 0.5224252
time: 278.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.9571065, 0.7206677, -0.9569757, 0.7205756, -1.4164405, 1.4156811
1: -1.0988159, 0.6236991, -1.0963949, 0.6220900, -0.8437334, 0.8559926
2: -3.6655307, -1.6739614, -3.6655347, -1.6741929, -1.3587922, 1.3589724
3: -4.1973038, -0.7901031, -4.1969810, -0.7901835, -1.4933171, 1.4937513
4: -5.0251608, -1.6261457, -5.0249438, -1.6272049, -2.1931751, 2.1903224
5: -4.3327398, -1.0026581, -4.3324909, -1.0026187, -1.4486284, 1.4490683
6: -8.4142494, -4.5946741, -8.4131680, -4.5954523, -1.4388468, 1.4416839
7: -4.6256547, -1.2472162, -4.6251593, -1.2472936, -2.1154394, 2.1147637
8: -0.1767009, 0.7755503, -0.1767210, 0.7754465, -0.8893811, 0.8893925
9: -1.5120796, 0.1959533, -1.5108706, 0.1949923, -1.1626205, 1.1663231

Time for backsubstitution: 4.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5227228, upper bound: 0.5223864
time: 245.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5227222, upper bound: 0.5224275
time: 198.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.9569133, 0.7206683, -0.9568220, 0.7207921, -1.4165837, 1.4156694
1: -1.0988135, 0.6227415, -1.0982889, 0.6209888, -0.8435833, 0.8592520
2: -3.6654830, -1.6739591, -3.6655064, -1.6739521, -1.3589182, 1.3589574
3: -4.1973014, -0.7902995, -4.1974311, -0.7904065, -1.4933362, 1.4939923
4: -5.0250106, -1.6261492, -5.0248108, -1.6266708, -2.1932020, 2.1902895
5: -4.3327093, -1.0028921, -4.3331718, -1.0028793, -1.4485501, 1.4497005
6: -8.4142408, -4.5950799, -8.4137468, -4.5959206, -1.4387733, 1.4418974
7: -4.6254163, -1.2472186, -4.6251326, -1.2472955, -2.1152191, 2.1147597
8: -0.1766253, 0.7755439, -0.1766606, 0.7758439, -0.8897486, 0.8893031
9: -1.5120745, 0.1954508, -1.5118809, 0.1944199, -1.1626611, 1.1670594

Time for backsubstitution: 4.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5227423, upper bound: 0.5223906
time: 74.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5227404, upper bound: 0.5224279
time: 101.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.9574500, 0.7198749, -0.9571675, 0.7200352, -1.4155889, 1.4157517
1: -1.0940381, 0.6258384, -1.0931517, 0.6242826, -0.8488121, 0.8454336
2: -3.6657772, -1.6739475, -3.6655266, -1.6741834, -1.3588296, 1.3588321
3: -4.1971092, -0.7899131, -4.1968985, -0.7900221, -1.4937129, 1.4930567
4: -5.0254474, -1.6284180, -5.0252905, -1.6288629, -2.1898341, 2.1909533
5: -4.3325605, -1.0024245, -4.3323870, -1.0025415, -1.4491763, 1.4482763
6: -8.4131441, -4.5941405, -8.4125795, -4.5945315, -1.4396689, 1.4391518
7: -4.6265068, -1.2472867, -4.6255736, -1.2473364, -2.1155648, 2.1152735
8: -0.1766727, 0.7755650, -0.1766812, 0.7754499, -0.8892621, 0.8894043
9: -1.5102959, 0.1968865, -1.5096850, 0.1962930, -1.1652217, 1.1622429

Time for backsubstitution: 4.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5225373, upper bound: 0.5226388
time: 74.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5225378, upper bound: 0.5226784
time: 31.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.9572570, 0.7198756, -0.9570140, 0.7202519, -1.4157324, 1.4157405
1: -1.0940359, 0.6248809, -1.0950449, 0.6231813, -0.8486623, 0.8486927
2: -3.6657298, -1.6739459, -3.6654987, -1.6739426, -1.3589559, 1.3588171
3: -4.1971059, -0.7901100, -4.1973491, -0.7902454, -1.4937322, 1.4932981
4: -5.0252962, -1.6284209, -5.0251589, -1.6283274, -2.1898620, 2.1909204
5: -4.3325291, -1.0026596, -4.3330669, -1.0028020, -1.4490979, 1.4489082
6: -8.4131336, -4.5945482, -8.4131575, -4.5949998, -1.4395959, 1.4393651
7: -4.6262693, -1.2472894, -4.6255469, -1.2473375, -2.1153450, 2.1152697
8: -0.1765969, 0.7755585, -0.1766209, 0.7758472, -0.8896297, 0.8893149
9: -1.5102913, 0.1963836, -1.5106955, 0.1957211, -1.1652627, 1.1629792

Time for backsubstitution: 4.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5225574, upper bound: 0.5226373
time: 204.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5225576, upper bound: 0.5226796
time: 32.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.9571159, 0.7208189, -0.9572263, 0.7207523, -1.4163826, 1.4160426
1: -1.1007100, 0.6237010, -1.0987216, 0.6242957, -0.8484945, 0.8549176
2: -3.6655364, -1.6739568, -3.6655488, -1.6741773, -1.3588419, 1.3589765
3: -4.1975279, -0.7901019, -4.1972518, -0.7900112, -1.4936814, 1.4937949
4: -5.0251713, -1.6257927, -5.0253739, -1.6267650, -2.1928542, 2.1909885
5: -4.3328137, -1.0026566, -4.3326302, -1.0025187, -1.4491267, 1.4489777
6: -8.4150648, -4.5946708, -8.4141369, -4.5945177, -1.4398096, 1.4418232
7: -4.6256695, -1.2471840, -4.6256537, -1.2472533, -2.1154265, 2.1153646
8: -0.1767140, 0.7755527, -0.1767450, 0.7754698, -0.8894194, 0.8894257
9: -1.5130523, 0.1959541, -1.5121340, 0.1963005, -1.1650131, 1.1658001

Time for backsubstitution: 4.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5227241, upper bound: 0.5226377
time: 188.50 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5227220, upper bound: 0.5226726
time: 318.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 511.85 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5225394, upper bound: 0.5223870
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5225374, upper bound: 0.5224283
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5225579, upper bound: 0.5223900
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5225579, upper bound: 0.5224252
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5227228, upper bound: 0.5223864
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5227222, upper bound: 0.5224275
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5227423, upper bound: 0.5223906
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5227404, upper bound: 0.5224279
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5225373, upper bound: 0.5226388
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5225378, upper bound: 0.5226784
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5225574, upper bound: 0.5226373
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5225576, upper bound: 0.5226796
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5227241, upper bound: 0.5226377
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 511.85
Output dim: 1, lower bound: -0.5227220, upper bound: 0.5226726
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 511.85
Output dim: 1, lower bound: -0.5227998, upper bound: 0.5227367
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 511.85
Output dim: 1, lower bound: -0.5226206, upper bound: 0.5225731
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 511.85
Output dim: 1, lower bound: -0.5226366, upper bound: 0.5225714
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 511.85
Output dim: 1, lower bound: -0.5228063, upper bound: 0.5225764
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 511.85
Output dim: 1, lower bound: -0.5228234, upper bound: 0.5225751
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 511.85
Output dim: 1, lower bound: -0.5226239, upper bound: 0.5225735
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 511.85
Output dim: 1, lower bound: -0.5226391, upper bound: 0.5228217
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 511.85
Output dim: 1, lower bound: -0.5228046, upper bound: 0.5228268
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 511.85
Output dim: 1, lower bound: -0.5228230, upper bound: 0.5228249
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=4, k_mid=3, eps_mid=0.0117188, abs_max=0.8648518323898315
rel_dist={1: [-0.5234064232429227, 0.5234093100156474]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2522

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945741, upper bound: 0.4945246
time: 114.49 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945763, upper bound: 0.4945757
time: 19.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 134.30 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 134.30
Output dim: 1, lower bound: -0.4945741, upper bound: 0.4945246
IS_A2, status: Status.UNKNOWN, split count: 1, time: 134.30
Output dim: 1, lower bound: -0.4945763, upper bound: 0.4945757

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.9573896, 0.7210121, -0.9574845, 0.7209600, -1.4108368, 1.4110441
1: -1.1015255, 0.6246668, -1.0994571, 0.6252006, -0.8438020, 0.8438982
2: -3.6656461, -1.6738957, -3.6656623, -1.6741300, -1.3452916, 1.3454731
3: -4.1976347, -0.7899210, -4.1973553, -0.7898378, -1.4638801, 1.4635121
4: -5.0255575, -1.6253605, -5.0257301, -1.6263264, -2.1714659, 2.1720948
5: -4.3330030, -1.0024294, -4.3328228, -1.0023072, -1.4188986, 1.4185245
6: -8.4153233, -4.5942554, -8.4143848, -4.5940952, -1.4034610, 1.4031928
7: -4.6257863, -1.2470965, -4.6257720, -1.2471650, -2.0936241, 2.0936770
8: -0.1768111, 0.7755800, -0.1768314, 0.7754993, -0.8870643, 0.8871383
9: -1.5134718, 0.1961916, -1.5125146, 0.1964598, -1.1561478, 1.1553240

Time for backsubstitution: 4.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2080

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945503, upper bound: 0.4942782
time: 19.65 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945501, upper bound: 0.4945046
time: 118.41 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.9584174, 0.7209653, -0.9584480, 0.7209657, -1.4114194, 1.4121025
1: -1.0996343, 0.6282543, -1.0996435, 0.6284162, -0.8493513, 0.8437125
2: -3.6658449, -1.6741171, -3.6658580, -1.6741168, -1.3455374, 1.3455622
3: -4.1973686, -0.7893600, -4.1973705, -0.7893325, -1.4641756, 1.4637892
4: -5.0270252, -1.6263089, -5.0270920, -1.6263075, -2.1721582, 2.1732221
5: -4.3328667, -1.0014926, -4.3328710, -1.0014671, -1.4196119, 1.4190115
6: -8.4143925, -4.5931339, -8.4143925, -4.5930862, -1.4049982, 1.4033346
7: -4.6258521, -1.2471557, -4.6258826, -1.2471552, -2.0939050, 2.0939307
8: -0.1771424, 0.7755325, -0.1771550, 0.7755340, -0.8874430, 0.8874924
9: -1.5127169, 0.1984509, -1.5127282, 0.1985135, -1.1588411, 1.1561596

Time for backsubstitution: 4.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2080

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945526, upper bound: 0.4943365
time: 17.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945530, upper bound: 0.4945554
time: 37.69 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 59.64 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 59.64
Output dim: 1, lower bound: -0.4945503, upper bound: 0.4942782
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 59.64
Output dim: 1, lower bound: -0.4945501, upper bound: 0.4945046
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 59.64
Output dim: 1, lower bound: -0.4945526, upper bound: 0.4943365
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 59.64
Output dim: 1, lower bound: -0.4945530, upper bound: 0.4945554

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.9573637, 0.7206944, -0.9572157, 0.7206076, -1.4104688, 1.4105145
1: -1.0991242, 0.6246601, -1.0966854, 0.6229903, -0.8406456, 0.8419410
2: -3.6656365, -1.6739063, -3.6656437, -1.6741517, -1.3452311, 1.3454392
3: -4.1973310, -0.7899257, -4.1970143, -0.7900121, -1.4636327, 1.4632237
4: -5.0255294, -1.6261172, -5.0252805, -1.6271589, -2.1707475, 2.1711135
5: -4.3327990, -1.0024362, -4.3325491, -1.0024141, -1.4184709, 1.4181937
6: -8.4142733, -4.5942621, -8.4132252, -4.5950317, -1.4028447, 1.4024687
7: -4.6257563, -1.2471876, -4.6252608, -1.2472701, -2.0934949, 2.0930977
8: -0.1767769, 0.7755726, -0.1767850, 0.7754704, -0.8869768, 0.8870579
9: -1.5122325, 0.1961891, -1.5110151, 0.1951490, -1.1544212, 1.1541471

Time for backsubstitution: 4.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4943555, upper bound: 0.4942558
time: 41.77 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944733, upper bound: 0.4942581
time: 90.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.9573736, 0.7208532, -0.9574668, 0.7207842, -1.4104040, 1.4108799
1: -1.1011361, 0.6246624, -1.0990264, 0.6251956, -0.8429769, 0.8408083
2: -3.6656415, -1.6739013, -3.6656585, -1.6741360, -1.3452809, 1.3454428
3: -4.1975818, -0.7899244, -4.1972966, -0.7898397, -1.4637656, 1.4632609
4: -5.0255408, -1.6256580, -5.0257106, -1.6266546, -2.1704297, 2.1717002
5: -4.3328867, -1.0024343, -4.3326969, -1.0023129, -1.4187760, 1.4180999
6: -8.4151812, -4.5942574, -8.4142284, -4.5940962, -1.4028555, 1.4025795
7: -4.6257710, -1.2471547, -4.6257572, -1.2472298, -2.0934825, 2.0936050
8: -0.1767909, 0.7755752, -0.1768091, 0.7754939, -0.8870158, 0.8870900
9: -1.5132643, 0.1961894, -1.5122854, 0.1964571, -1.1557399, 1.1535933

Time for backsubstitution: 4.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4943522, upper bound: 0.4944328
time: 22.85 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944733, upper bound: 0.4944281
time: 100.57 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.9583914, 0.7206475, -0.9581794, 0.7206131, -1.4110516, 1.4115732
1: -1.0972332, 0.6282476, -1.0968723, 0.6262057, -0.8461949, 0.8417556
2: -3.6658356, -1.6741269, -3.6658392, -1.6741381, -1.3454769, 1.3455279
3: -4.1970639, -0.7893636, -4.1970277, -0.7895074, -1.4639286, 1.4635005
4: -5.0269985, -1.6270647, -5.0266423, -1.6271399, -2.1714401, 2.1722398
5: -4.3326626, -1.0014991, -4.3325963, -1.0015731, -1.4191844, 1.4186805
6: -8.4133434, -4.5931392, -8.4132347, -4.5940256, -1.4043816, 1.4026102
7: -4.6258230, -1.2472472, -4.6253710, -1.2472609, -2.0937762, 2.0933509
8: -0.1771080, 0.7755250, -0.1771086, 0.7755052, -0.8873560, 0.8874122
9: -1.5114782, 0.1984483, -1.5112288, 0.1972027, -1.1571145, 1.1549829

Time for backsubstitution: 4.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4943561, upper bound: 0.4943175
time: 29.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944758, upper bound: 0.4943095
time: 21.24 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.9584014, 0.7208064, -0.9584305, 0.7207900, -1.4109874, 1.4119378
1: -1.0992444, 0.6282501, -1.0992129, 0.6284115, -0.8485255, 0.8406230
2: -3.6658409, -1.6741222, -3.6658530, -1.6741219, -1.3455271, 1.3455317
3: -4.1973143, -0.7893617, -4.1973095, -0.7893345, -1.4640615, 1.4635375
4: -5.0270090, -1.6266053, -5.0270729, -1.6266365, -2.1711230, 2.1728263
5: -4.3327522, -1.0014979, -4.3327441, -1.0014726, -1.4194895, 1.4185870
6: -8.4142523, -4.5931349, -8.4142389, -4.5930896, -1.4043922, 1.4027215
7: -4.6258383, -1.2472140, -4.6258669, -1.2472198, -2.0937629, 2.0938592
8: -0.1771219, 0.7755275, -0.1771325, 0.7755287, -0.8873950, 0.8874444
9: -1.5125096, 0.1984491, -1.5124986, 0.1985112, -1.1584324, 1.1544287

Time for backsubstitution: 4.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2981

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4943563, upper bound: 0.4944787
time: 13.99 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944770, upper bound: 0.4944803
time: 13.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 32.34 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.34
Output dim: 1, lower bound: -0.4943555, upper bound: 0.4942558
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.34
Output dim: 1, lower bound: -0.4944733, upper bound: 0.4942581
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.34
Output dim: 1, lower bound: -0.4943522, upper bound: 0.4944328
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.34
Output dim: 1, lower bound: -0.4944733, upper bound: 0.4944281
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.34
Output dim: 1, lower bound: -0.4943561, upper bound: 0.4943175
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.34
Output dim: 1, lower bound: -0.4944758, upper bound: 0.4943095
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.34
Output dim: 1, lower bound: -0.4943563, upper bound: 0.4944787
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.34
Output dim: 1, lower bound: -0.4944770, upper bound: 0.4944803

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.9576927, 0.7196344, -0.9571478, 0.7197075, -1.4095941, 1.4093363
1: -1.0919695, 0.6267964, -1.0903471, 0.6229750, -0.8312832, 0.8331732
2: -3.6658773, -1.6739469, -3.6656189, -1.6741958, -1.3451934, 1.3452537
3: -4.1968536, -0.7897364, -4.1965847, -0.7900256, -1.4629128, 1.4625016
4: -5.0258012, -1.6291285, -5.0251865, -1.6297333, -2.1675935, 2.1677248
5: -4.3324833, -1.0022072, -4.3322344, -1.0024403, -1.4179292, 1.4175563
6: -8.4120989, -4.5937343, -8.4113188, -4.5950508, -1.3999138, 1.4000993
7: -4.6265860, -1.2473432, -4.6251669, -1.2474096, -2.0936406, 2.0927472
8: -0.1767240, 0.7755833, -0.1767067, 0.7754472, -0.8868728, 0.8870044
9: -1.5092682, 0.1971200, -1.5082234, 0.1951399, -1.1511574, 1.1509470

Time for backsubstitution: 4.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4942008, upper bound: 0.4941320
time: 97.79 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4942200, upper bound: 0.4941307
time: 13.10 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.9573588, 0.7206571, -0.9572120, 0.7205777, -1.4104300, 1.4096277
1: -1.0987120, 0.6246586, -1.0963548, 0.6229893, -0.8295355, 0.8419175
2: -3.6656353, -1.6739560, -3.6656435, -1.6741917, -1.3452160, 1.3454007
3: -4.1973004, -0.7899270, -4.1969895, -0.7900131, -1.4627268, 1.4631891
4: -5.0255251, -1.6261650, -5.0252767, -1.6271973, -2.1706963, 2.1677437
5: -4.3327608, -1.0024388, -4.3325181, -1.0024146, -1.4176944, 1.4181623
6: -8.4142046, -4.5942636, -8.4131689, -4.5950336, -1.3995485, 1.4024616
7: -4.6257482, -1.2472190, -4.6252542, -1.2472947, -2.0934811, 2.0927958
8: -0.1767659, 0.7755721, -0.1767761, 0.7754700, -0.8870335, 0.8870315
9: -1.5120437, 0.1961878, -1.5108637, 0.1951483, -1.1502445, 1.1540829

Time for backsubstitution: 4.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4943254, upper bound: 0.4941298
time: 249.93 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4943407, upper bound: 0.4941309
time: 19.05 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.9577029, 0.7198740, -0.9573990, 0.7199737, -1.4094992, 1.4097040
1: -1.0940317, 0.6267985, -1.0927392, 0.6251808, -0.8346993, 0.8307130
2: -3.6658826, -1.6739423, -3.6656334, -1.6741803, -1.3452435, 1.3452578
3: -4.1971178, -0.7897367, -4.1968846, -0.7898537, -1.4631371, 1.4624389
4: -5.0258117, -1.6284128, -5.0256166, -1.6290208, -2.1671367, 2.1684074
5: -4.3325849, -1.0022049, -4.3323956, -1.0023403, -1.4182532, 1.4173336
6: -8.4131441, -4.5937290, -8.4124737, -4.5941153, -1.4004049, 1.3997010
7: -4.6266026, -1.2472873, -4.6256633, -1.2473428, -2.0935903, 2.0933042
8: -0.1767383, 0.7755868, -0.1767310, 0.7754718, -0.8869128, 0.8870381
9: -1.5103092, 0.1971209, -1.5095077, 0.1964483, -1.1529063, 1.1497653

Time for backsubstitution: 4.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4942057, upper bound: 0.4942971
time: 13.18 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4942205, upper bound: 0.4943012
time: 24.09 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.9573689, 0.7208160, -0.9574625, 0.7207545, -1.4103652, 1.4099938
1: -1.1007035, 0.6246612, -1.0986795, 0.6251950, -0.8343213, 0.8407850
2: -3.6656399, -1.6739509, -3.6656568, -1.6741753, -1.3452660, 1.3454045
3: -4.1975346, -0.7899252, -4.1972589, -0.7898409, -1.4631016, 1.4632258
4: -5.0255356, -1.6257951, -5.0257068, -1.6267651, -2.1703796, 2.1684186
5: -4.3328371, -1.0024357, -4.3326554, -1.0023147, -1.4182034, 1.4180686
6: -8.4150639, -4.5942602, -8.4141340, -4.5940976, -1.4005265, 1.4025726
7: -4.6257644, -1.2471855, -4.6257491, -1.2472537, -2.0934677, 2.0934005
8: -0.1767797, 0.7755746, -0.1768002, 0.7754934, -0.8870727, 0.8870640
9: -1.5130659, 0.1961887, -1.5121268, 0.1964566, -1.1526701, 1.1535290

Time for backsubstitution: 4.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4943239, upper bound: 0.4942955
time: 24.11 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4943405, upper bound: 0.4942971
time: 70.63 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.9587203, 0.7195877, -0.9581113, 0.7197131, -1.4101774, 1.4103947
1: -1.0900785, 0.6303838, -1.0905341, 0.6261907, -0.8368321, 0.8329881
2: -3.6660759, -1.6741679, -3.6658144, -1.6741822, -1.3454397, 1.3453426
3: -4.1965861, -0.7891742, -4.1966004, -0.7895193, -1.4632081, 1.4627789
4: -5.0272694, -1.6300774, -5.0265489, -1.6297138, -2.1682858, 2.1688519
5: -4.3323469, -1.0012702, -4.3322811, -1.0015996, -1.4186419, 1.4180436
6: -8.4111662, -4.5926123, -8.4113283, -4.5940423, -1.4014505, 1.4002407
7: -4.6266537, -1.2474022, -4.6252766, -1.2473989, -2.0939217, 2.0930004
8: -0.1770555, 0.7755358, -0.1770303, 0.7754821, -0.8872518, 0.8873588
9: -1.5085132, 0.1993797, -1.5084374, 0.1971940, -1.1538506, 1.1517832

Time for backsubstitution: 4.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4942087, upper bound: 0.4941819
time: 14.97 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4942228, upper bound: 0.4941811
time: 231.75 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.9583867, 0.7206103, -0.9581755, 0.7205833, -1.4110134, 1.4106861
1: -1.0968206, 0.6282465, -1.0965416, 0.6262047, -0.8350853, 0.8417320
2: -3.6658344, -1.6741769, -3.6658387, -1.6741769, -1.3454621, 1.3454893
3: -4.1970339, -0.7893640, -4.1970034, -0.7895069, -1.4630222, 1.4634664
4: -5.0269928, -1.6271129, -5.0266380, -1.6271782, -2.1713886, 2.1688700
5: -4.3326254, -1.0015010, -4.3325663, -1.0015746, -1.4184078, 1.4186494
6: -8.4132748, -4.5931420, -8.4131813, -4.5940256, -1.4010849, 1.4026037
7: -4.6258154, -1.2472783, -4.6253648, -1.2472849, -2.0937614, 2.0930500
8: -0.1770974, 0.7755246, -0.1770999, 0.7755049, -0.8874129, 0.8873860
9: -1.5112894, 0.1984476, -1.5110774, 0.1972018, -1.1529379, 1.1549182

Time for backsubstitution: 4.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4943286, upper bound: 0.4941794
time: 11.55 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4943446, upper bound: 0.4941814
time: 183.81 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.9587304, 0.7198275, -0.9583624, 0.7199795, -1.4100827, 1.4107624
1: -1.0921407, 0.6303859, -1.0929254, 0.6283963, -0.8402475, 0.8305283
2: -3.6660812, -1.6741629, -3.6658282, -1.6741664, -1.3454890, 1.3453466
3: -4.1968513, -0.7891736, -4.1968975, -0.7893471, -1.4634330, 1.4627165
4: -5.0272794, -1.6293617, -5.0269780, -1.6290017, -2.1678300, 2.1695337
5: -4.3324490, -1.0012683, -4.3324423, -1.0014985, -1.4189663, 1.4178209
6: -8.4122114, -4.5926075, -8.4124832, -4.5931072, -1.4019420, 1.3998432
7: -4.6266685, -1.2473458, -4.6257725, -1.2473328, -2.0938718, 2.0935583
8: -0.1770696, 0.7755393, -0.1770546, 0.7755066, -0.8872920, 0.8873924
9: -1.5095545, 0.1993804, -1.5097212, 0.1985025, -1.1555996, 1.1506014

Time for backsubstitution: 4.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4942073, upper bound: 0.4943463
time: 34.93 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4942238, upper bound: 0.4943467
time: 236.71 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.9583966, 0.7207689, -0.9584265, 0.7207603, -1.4109483, 1.4110521
1: -1.0988116, 0.6282489, -1.0988663, 0.6284102, -0.8398703, 0.8406000
2: -3.6658399, -1.6741720, -3.6658523, -1.6741618, -1.3455124, 1.3454928
3: -4.1972704, -0.7893623, -4.1972737, -0.7893348, -1.4633973, 1.4635031
4: -5.0270038, -1.6267431, -5.0270686, -1.6267461, -2.1710715, 2.1695454
5: -4.3327017, -1.0015004, -4.3327036, -1.0014745, -1.4189167, 1.4185557
6: -8.4141321, -4.5931368, -8.4141445, -4.5930915, -1.4020631, 1.4027147
7: -4.6258316, -1.2472450, -4.6258597, -1.2472447, -2.0937479, 2.0936546
8: -0.1771113, 0.7755271, -0.1771240, 0.7755283, -0.8874518, 0.8874182
9: -1.5123115, 0.1984483, -1.5123403, 0.1985106, -1.1553634, 1.1543653

Time for backsubstitution: 4.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3091
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2931
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2493

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4943288, upper bound: 0.4943460
time: 221.53 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4943427, upper bound: 0.4943434
time: 18.22 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 244.32 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4942008, upper bound: 0.4941320
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4942200, upper bound: 0.4941307
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4943254, upper bound: 0.4941298
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4943407, upper bound: 0.4941309
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4942057, upper bound: 0.4942971
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4942205, upper bound: 0.4943012
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4943239, upper bound: 0.4942955
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4943405, upper bound: 0.4942971
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4942087, upper bound: 0.4941819
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4942228, upper bound: 0.4941811
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4943286, upper bound: 0.4941794
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4943446, upper bound: 0.4941814
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4942073, upper bound: 0.4943463
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4942238, upper bound: 0.4943467
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4943288, upper bound: 0.4943460
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 244.32
Output dim: 1, lower bound: -0.4943427, upper bound: 0.4943434

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.9584613, 0.7198218, -0.9580607, 0.7199734, -1.4097638, 1.4104139
1: -1.0921276, 0.6293651, -1.0929110, 0.6272469, -0.8390822, 0.8294851
2: -3.6659703, -1.6741693, -3.6657047, -1.6741743, -1.3453640, 1.3452097
3: -4.1968384, -0.7893625, -4.1968851, -0.7895589, -1.4632641, 1.4625636
4: -5.0268917, -1.6293750, -5.0265427, -1.6290160, -2.1674898, 2.1691563
5: -4.3324165, -1.0015022, -4.3324080, -1.0017622, -1.4186614, 1.4175396
6: -8.4122047, -4.5930462, -8.4124756, -4.5935993, -1.4015510, 1.3994931
7: -4.6265678, -1.2473488, -4.6256595, -1.2473354, -2.0937548, 2.0934284
8: -0.1769984, 0.7755158, -0.1769748, 0.7754803, -0.8871719, 0.8872657
9: -1.5095290, 0.1991306, -1.5096930, 0.1982214, -1.1553661, 1.1503904

Time for backsubstitution: 4.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4941705, upper bound: 0.4942758
time: 18.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4941706, upper bound: 0.4943089
time: 457.48 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.9582589, 0.7198228, -0.9579075, 0.7201898, -1.4099040, 1.4104011
1: -1.0921240, 0.6283593, -1.0948040, 0.6261454, -0.8389270, 0.8327314
2: -3.6659198, -1.6741673, -3.6656764, -1.6739330, -1.3454869, 1.3451934
3: -4.1968374, -0.7895688, -4.1973372, -0.7897828, -1.4632828, 1.4627907
4: -5.0267344, -1.6293783, -5.0264101, -1.6284809, -2.1675014, 2.1691236
5: -4.3323841, -1.0017493, -4.3330884, -1.0020219, -1.4185756, 1.4181658
6: -8.4121971, -4.5934715, -8.4130516, -4.5940661, -1.4014730, 1.3996929
7: -4.6263161, -1.2473512, -4.6256318, -1.2473379, -2.0935233, 2.0934238
8: -0.1769190, 0.7755092, -0.1769143, 0.7758776, -0.8875354, 0.8871750
9: -1.5095240, 0.1986048, -1.5107032, 0.1976494, -1.1553946, 1.1510975

Time for backsubstitution: 4.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4941849, upper bound: 0.4942785
time: 665.28 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4941865, upper bound: 0.4943096
time: 17.51 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.9581276, 0.7207640, -0.9581250, 0.7207540, -1.4106302, 1.4107037
1: -1.0987988, 0.6272277, -1.0988518, 0.6272609, -0.8387055, 0.8395563
2: -3.6657279, -1.6741784, -3.6657281, -1.6741697, -1.3453871, 1.3453565
3: -4.1972585, -0.7895507, -4.1972618, -0.7895468, -1.4632281, 1.4633501
4: -5.0266161, -1.6267563, -5.0266323, -1.6267617, -2.1707318, 2.1691675
5: -4.3326712, -1.0017328, -4.3326683, -1.0017374, -1.4186114, 1.4182749
6: -8.4141245, -4.5935740, -8.4141350, -4.5935826, -1.4016721, 1.4023645
7: -4.6257291, -1.2472464, -4.6257463, -1.2472470, -2.0936317, 2.0935247
8: -0.1770402, 0.7755036, -0.1770438, 0.7755021, -0.8873314, 0.8872916
9: -1.5122859, 0.1981985, -1.5123118, 0.1982293, -1.1551304, 1.1541531

Time for backsubstitution: 4.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3091
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2931
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3109

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4942927, upper bound: 0.4942849
time: 28.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4942936, upper bound: 0.4943081
time: 85.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 118.76 seconds
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 118.76
Output dim: 1, lower bound: -0.4941705, upper bound: 0.4942758
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 118.76
Output dim: 1, lower bound: -0.4941706, upper bound: 0.4943089
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 118.76
Output dim: 1, lower bound: -0.4941849, upper bound: 0.4942785
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 118.76
Output dim: 1, lower bound: -0.4941865, upper bound: 0.4943096
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 118.76
Output dim: 1, lower bound: -0.4942927, upper bound: 0.4942849
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 118.76
Output dim: 1, lower bound: -0.4942936, upper bound: 0.4943081
Binary search (step 2): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.8498140573501587
rel_dist={1: [-0.49469755591069386, 0.49469970537319075]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 11916.74 seconds
