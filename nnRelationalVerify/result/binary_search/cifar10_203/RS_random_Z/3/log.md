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
execution time: IAR + LP analysis = 5.59 + 120.95 = 126.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.6552173, upper bound: 0.6552176


# Binary Search by BASE starts (time budget: 17873.46 seconds, max iter: 100)

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
Binary search time: 209.50 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Relational Split (RS_random_Z) starts
Time budget: 17663.96 seconds

## Binary search (step 0) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2080

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2382

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788605, upper bound: 0.5788535
time: 41.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788550, upper bound: 0.5788576
time: 67.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 109.05 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 109.05
Output dim: 1, lower bound: -0.5788605, upper bound: 0.5788535
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 109.05
Output dim: 1, lower bound: -0.5788550, upper bound: 0.5788576

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311640, 1.4311618
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949125, 0.8949093
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867106, 1.3867166
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552926, 1.5555494
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2416959, 2.2417564
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5121193, 1.5123795
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5232632, 1.5233742
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1592567, 2.1594796
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949400, 0.8949409
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962769, 1.1962862

Time for backsubstitution: 4.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3087

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3589

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788589, upper bound: 0.5788473
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788589, upper bound: 0.5788512
time: 7.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311621, 1.4311640
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949094, 0.8949126
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867166, 1.3867106
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5555491, 1.5552926
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2417564, 2.2416956
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5123794, 1.5121193
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5233743, 1.5232633
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1594799, 2.1592565
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949410, 0.8949399
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962864, 1.1962768

Time for backsubstitution: 4.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2151

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3589

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788533, upper bound: 0.5788612
time: 42.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788533, upper bound: 0.5788610
time: 140.91 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 188.07 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 188.07
Output dim: 1, lower bound: -0.5788589, upper bound: 0.5788473
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 188.07
Output dim: 1, lower bound: -0.5788589, upper bound: 0.5788512
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 188.07
Output dim: 1, lower bound: -0.5788533, upper bound: 0.5788612
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 188.07
Output dim: 1, lower bound: -0.5788533, upper bound: 0.5788610

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311640, 1.4311618
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949125, 0.8949093
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867106, 1.3867166
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552926, 1.5555494
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2416959, 2.2417564
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5121193, 1.5123795
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5232632, 1.5233742
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1592567, 2.1594796
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949400, 0.8949409
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962769, 1.1962862

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 464

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788610, upper bound: 0.5788500
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788610, upper bound: 0.5788511
time: 9.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311640, 1.4311618
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949125, 0.8949093
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867106, 1.3867166
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552926, 1.5555494
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2416959, 2.2417564
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5121193, 1.5123795
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5232632, 1.5233742
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1592567, 2.1594796
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949400, 0.8949409
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962769, 1.1962862

Time for backsubstitution: 4.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785198, upper bound: 0.5784976
time: 16.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785016, upper bound: 0.5785175
time: 7.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311621, 1.4311640
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949094, 0.8949126
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867166, 1.3867106
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5555491, 1.5552926
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2417564, 2.2416956
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5123794, 1.5121193
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5233743, 1.5232633
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1594799, 2.1592565
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949410, 0.8949399
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962864, 1.1962768

Time for backsubstitution: 4.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2049

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2931

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787720, upper bound: 0.5787779
time: 20.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787720, upper bound: 0.5787790
time: 20.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311621, 1.4311640
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949094, 0.8949126
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867166, 1.3867106
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5555491, 1.5552926
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2417564, 2.2416956
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5123794, 1.5121193
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5233743, 1.5232633
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1594799, 2.1592565
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949410, 0.8949399
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962864, 1.1962768

Time for backsubstitution: 4.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3368

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3346

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788385, upper bound: 0.5787560
time: 22.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787500, upper bound: 0.5788440
time: 11.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 38.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.40
Output dim: 1, lower bound: -0.5788610, upper bound: 0.5788500
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.40
Output dim: 1, lower bound: -0.5788610, upper bound: 0.5788511
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.40
Output dim: 1, lower bound: -0.5785198, upper bound: 0.5784976
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.40
Output dim: 1, lower bound: -0.5785016, upper bound: 0.5785175
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.40
Output dim: 1, lower bound: -0.5787720, upper bound: 0.5787779
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.40
Output dim: 1, lower bound: -0.5787720, upper bound: 0.5787790
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.40
Output dim: 1, lower bound: -0.5788385, upper bound: 0.5787560
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.40
Output dim: 1, lower bound: -0.5787500, upper bound: 0.5788440

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311640, 1.4311618
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949125, 0.8949093
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867106, 1.3867166
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552926, 1.5555494
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2416959, 2.2417564
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5121193, 1.5123795
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5232632, 1.5233742
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1592567, 2.1594796
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949400, 0.8949409
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962769, 1.1962862

Time for backsubstitution: 4.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788443, upper bound: 0.5788423
time: 20.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788489, upper bound: 0.5788345
time: 7.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311640, 1.4311618
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949125, 0.8949093
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867106, 1.3867166
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552926, 1.5555494
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2416959, 2.2417564
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5121193, 1.5123795
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5232632, 1.5233742
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1592567, 2.1594796
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949400, 0.8949409
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962769, 1.1962862

Time for backsubstitution: 4.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2474

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788345, upper bound: 0.5788286
time: 218.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788345, upper bound: 0.5788262
time: 18.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311486, 1.4311389
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948711, 0.8948690
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3864970, 1.3865404
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5540164, 1.5543096
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2397361, 2.2405567
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5106039, 1.5109081
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227044, 1.5228089
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1575360, 2.1581502
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949153, 0.8949163
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962765, 1.1962861

Time for backsubstitution: 4.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2483

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2681

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783923, upper bound: 0.5783692
time: 191.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783927, upper bound: 0.5783665
time: 13.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311410, 1.4311464
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948722, 0.8948680
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865345, 1.3865030
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5540528, 1.5542731
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2404962, 2.2397971
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5106478, 1.5108641
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5226982, 1.5228151
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1579270, 2.1577592
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949153, 0.8949163
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962765, 1.1962858

Time for backsubstitution: 4.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3347

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 646

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784159, upper bound: 0.5785132
time: 188.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785017, upper bound: 0.5784313
time: 7.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311621, 1.4311640
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949092, 0.8949124
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867161, 1.3867105
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5555491, 1.5552924
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2417560, 2.2416952
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5123794, 1.5121193
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5233741, 1.5232626
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1594796, 2.1592567
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949409, 0.8949399
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962860, 1.1962767

Time for backsubstitution: 4.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3071

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2421

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787662, upper bound: 0.5786741
time: 55.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786671, upper bound: 0.5787736
time: 28.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311621, 1.4311640
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949092, 0.8949126
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867166, 1.3867103
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5555491, 1.5552926
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2417564, 2.2416952
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5123794, 1.5121193
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5233736, 1.5232633
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1594799, 2.1592565
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949409, 0.8949399
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962864, 1.1962764

Time for backsubstitution: 4.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2265

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 427

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786126, upper bound: 0.5787702
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787644, upper bound: 0.5786183
time: 7.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4316334, 1.4316055
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948860, 0.8948792
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3868322, 1.3867975
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5553001, 1.5550363
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2419848, 2.2419147
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5120616, 1.5117791
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5222256, 1.5220374
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1594830, 2.1592598
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948961, 0.8949073
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962420, 1.1962126

Time for backsubstitution: 4.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2519

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788394, upper bound: 0.5787548
time: 63.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788394, upper bound: 0.5787577
time: 60.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4316033, 1.4316355
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948760, 0.8948890
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3868036, 1.3868263
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552930, 1.5550432
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2419753, 2.2419243
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5120392, 1.5118014
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5221484, 1.5221148
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1594830, 2.1592598
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949083, 0.8948953
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962219, 1.1962327

Time for backsubstitution: 4.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2373

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785869, upper bound: 0.5786833
time: 17.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785869, upper bound: 0.5786825
time: 16.92 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 38.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5788443, upper bound: 0.5788423
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5788489, upper bound: 0.5788345
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5788345, upper bound: 0.5788286
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5788345, upper bound: 0.5788262
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5783923, upper bound: 0.5783692
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5783927, upper bound: 0.5783665
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5784159, upper bound: 0.5785132
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5785017, upper bound: 0.5784313
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5787662, upper bound: 0.5786741
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5786671, upper bound: 0.5787736
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5786126, upper bound: 0.5787702
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5787644, upper bound: 0.5786183
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5788394, upper bound: 0.5787548
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5788394, upper bound: 0.5787577
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5785869, upper bound: 0.5786833
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.62
Output dim: 1, lower bound: -0.5785869, upper bound: 0.5786825

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311526, 1.4311544
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948905, 0.8948789
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3866749, 1.3866367
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552497, 1.5554643
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2416973, 2.2417541
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5120630, 1.5122817
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5232408, 1.5233122
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1592689, 2.1594458
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949381, 0.8949386
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962589, 1.1962259

Time for backsubstitution: 4.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2288

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3065

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788316, upper bound: 0.5786697
time: 7.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786736, upper bound: 0.5788310
time: 39.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311564, 1.4311504
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948820, 0.8948874
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3866308, 1.3866808
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552073, 1.5555066
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2416935, 2.2417581
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5120213, 1.5123231
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5232012, 1.5233518
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1592226, 2.1594920
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949376, 0.8949391
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962165, 1.1962682

Time for backsubstitution: 4.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2950

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3090

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786985, upper bound: 0.5787099
time: 8.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787258, upper bound: 0.5786842
time: 42.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311640, 1.4311618
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949125, 0.8949093
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867106, 1.3867166
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552926, 1.5555494
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2416959, 2.2417564
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5121193, 1.5123795
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5232632, 1.5233742
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1592567, 2.1594796
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949400, 0.8949409
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962769, 1.1962862

Time for backsubstitution: 4.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2216

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788271, upper bound: 0.5787387
time: 16.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787444, upper bound: 0.5788170
time: 32.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311640, 1.4311618
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949125, 0.8949093
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867106, 1.3867166
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552926, 1.5555494
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2416959, 2.2417564
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5121193, 1.5123795
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5232632, 1.5233742
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1592567, 2.1594796
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949400, 0.8949409
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962769, 1.1962862

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3458

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2421

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788266, upper bound: 0.5787199
time: 14.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787252, upper bound: 0.5788151
time: 32.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311377, 1.4311291
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948476, 0.8948436
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3864563, 1.3865008
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5540129, 1.5543065
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2395701, 2.2404137
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5105703, 1.5108762
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5226117, 1.5227213
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1574812, 2.1580939
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949143, 0.8949152
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962545, 1.1962630

Time for backsubstitution: 4.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2453

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5782178, upper bound: 0.5781799
time: 16.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5782082, upper bound: 0.5781933
time: 45.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311386, 1.4311280
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948457, 0.8948456
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3864577, 1.3864995
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5540131, 1.5543063
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2395935, 2.2403903
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5105722, 1.5108743
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5226165, 1.5227165
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1574798, 2.1580954
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949142, 0.8949153
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962533, 1.1962639

Time for backsubstitution: 4.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 3379

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 472

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783631, upper bound: 0.5783163
time: 35.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783398, upper bound: 0.5783411
time: 49.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311495, 1.4311552
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948708, 0.8948666
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865223, 1.3864914
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5540533, 1.5542741
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2404990, 2.2398007
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5106380, 1.5108551
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5226567, 1.5227764
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1579289, 2.1577611
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949105, 0.8949113
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962707, 1.1962799

Time for backsubstitution: 4.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5781240, upper bound: 0.5783362
time: 8.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5782356, upper bound: 0.5781369
time: 106.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311500, 1.4311547
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948708, 0.8948666
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865230, 1.3864908
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5540538, 1.5542737
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2404995, 2.2398005
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5106390, 1.5108542
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5226595, 1.5227739
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1579289, 2.1577611
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949104, 0.8949116
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962707, 1.1962799

Time for backsubstitution: 4.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2300

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2513

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779919, upper bound: 0.5779181
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5779919, upper bound: 0.5779181
time: 8.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311649, 1.4311676
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949525, 0.8949331
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3856677, 1.3857641
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5444450, 1.5449319
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2281873, 2.2289996
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5004208, 1.5009546
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5170542, 1.5171524
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1504171, 2.1506891
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948706, 0.8948659
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962655, 1.1962558

Time for backsubstitution: 4.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2991

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785487, upper bound: 0.5784470
time: 181.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785394, upper bound: 0.5784553
time: 11.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311653, 1.4311671
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949299, 0.8949558
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3857697, 1.3856621
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5451889, 1.5441883
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2290599, 2.2281265
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5012150, 1.5001605
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5172640, 1.5169429
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1509120, 2.1501942
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948670, 0.8948695
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962650, 1.1962563

Time for backsubstitution: 4.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3295

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2305

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5780713, upper bound: 0.5783140
time: 40.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5782089, upper bound: 0.5781714
time: 17.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4215618, 1.4209094
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8935497, 0.8937190
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3841485, 1.3839717
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5461124, 1.5452905
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2423294, 2.2422841
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5048113, 1.5039418
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5121084, 1.5113010
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1587439, 2.1585348
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949766, 0.8949859
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1972172, 1.1975015

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785713, upper bound: 0.5787588
time: 7.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786042, upper bound: 0.5787273
time: 7.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4209071, 1.4215641
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8937156, 0.8935531
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3839775, 1.3841424
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5455469, 1.5458560
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2423456, 2.2422683
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5042019, 1.5045514
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5114108, 1.5119983
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1587582, 2.1585205
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949867, 0.8949757
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1975114, 1.1972071

Time for backsubstitution: 4.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2674

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3494

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787630, upper bound: 0.5786212
time: 7.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787630, upper bound: 0.5786212
time: 8.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4316334, 1.4316055
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948860, 0.8948792
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3868322, 1.3867975
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5553001, 1.5550363
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2419848, 2.2419147
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5120616, 1.5117791
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5222256, 1.5220374
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1594830, 2.1592598
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948961, 0.8949073
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962420, 1.1962126

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3090

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 754

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787966, upper bound: 0.5787120
time: 23.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787933, upper bound: 0.5787126
time: 241.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4316334, 1.4316055
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948860, 0.8948792
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3868322, 1.3867975
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5553001, 1.5550363
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2419848, 2.2419147
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5120616, 1.5117791
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5222256, 1.5220374
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1594830, 2.1592598
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948961, 0.8949073
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962420, 1.1962126

Time for backsubstitution: 4.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 688

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788239, upper bound: 0.5787385
time: 30.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788238, upper bound: 0.5787374
time: 29.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4315882, 1.4316187
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8947951, 0.8948591
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867667, 1.3867712
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5534714, 1.5537913
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2416501, 2.2416883
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5102036, 1.5105177
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5210626, 1.5213559
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1580539, 2.1582513
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949033, 0.8948908
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1961612, 1.1961838

Time for backsubstitution: 4.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3345

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784730, upper bound: 0.5785217
time: 26.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784226, upper bound: 0.5785696
time: 14.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4315865, 1.4316204
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948461, 0.8948081
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3867483, 1.3867894
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5540408, 1.5532217
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2417397, 2.2415986
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5107558, 1.5099659
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5213892, 1.5210292
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1584744, 2.1578312
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949040, 0.8948902
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1961733, 1.1961719

Time for backsubstitution: 4.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2454

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784593, upper bound: 0.5785255
time: 20.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784310, upper bound: 0.5785558
time: 10.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 35.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5788316, upper bound: 0.5786697
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5786736, upper bound: 0.5788310
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5786985, upper bound: 0.5787099
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5787258, upper bound: 0.5786842
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5788271, upper bound: 0.5787387
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5787444, upper bound: 0.5788170
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5788266, upper bound: 0.5787199
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5787252, upper bound: 0.5788151
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5782178, upper bound: 0.5781799
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5782082, upper bound: 0.5781933
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5783631, upper bound: 0.5783163
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5783398, upper bound: 0.5783411
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5781240, upper bound: 0.5783362
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5782356, upper bound: 0.5781369
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5779919, upper bound: 0.5779181
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5779919, upper bound: 0.5779181
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5785487, upper bound: 0.5784470
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5785394, upper bound: 0.5784553
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5780713, upper bound: 0.5783140
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5782089, upper bound: 0.5781714
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5785713, upper bound: 0.5787588
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5786042, upper bound: 0.5787273
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5787630, upper bound: 0.5786212
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5787630, upper bound: 0.5786212
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5787966, upper bound: 0.5787120
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5787933, upper bound: 0.5787126
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5788239, upper bound: 0.5787385
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5788238, upper bound: 0.5787374
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5784730, upper bound: 0.5785217
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5784226, upper bound: 0.5785696
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5784593, upper bound: 0.5785255
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 35.31
Output dim: 1, lower bound: -0.5784310, upper bound: 0.5785558

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4308791, 1.4308748
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8943413, 0.8943073
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3861678, 1.3861713
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5253584, 1.5264983
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2324057, 2.2327447
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4815943, 1.4827911
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5082736, 1.5088674
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1376519, 2.1386309
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948090, 0.8948133
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1959609, 1.1959417

Time for backsubstitution: 4.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3098

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787607, upper bound: 0.5785982
time: 37.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787582, upper bound: 0.5785988
time: 11.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4308734, 1.4308805
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8943189, 0.8943297
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3862092, 1.3861295
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5262840, 1.5255728
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2326880, 2.2324624
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4825723, 1.4818127
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5087962, 1.5083450
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1384542, 2.1378288
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948128, 0.8948095
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1959747, 1.1959279

Time for backsubstitution: 4.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2485

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785646, upper bound: 0.5786429
time: 60.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784863, upper bound: 0.5787221
time: 42.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311103, 1.4311048
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8908958, 0.8909284
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3861055, 1.3861604
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5551610, 1.5554564
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2410522, 2.2411075
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5114771, 1.5117819
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5206943, 1.5208549
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1583478, 2.1586227
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948619, 0.8948628
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1909238, 1.1910095

Time for backsubstitution: 4.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 840

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786824, upper bound: 0.5778866
time: 27.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5778753, upper bound: 0.5786958
time: 40.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311107, 1.4311044
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8909230, 0.8909010
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3861108, 1.3861554
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5551577, 1.5554599
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2410426, 2.2411172
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5114800, 1.5117791
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5207043, 1.5208447
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1583536, 2.1586170
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948612, 0.8948635
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1909577, 1.1909754

Time for backsubstitution: 4.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2397

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786667, upper bound: 0.5786463
time: 55.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786853, upper bound: 0.5786261
time: 41.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311550, 1.4311512
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949516, 0.8949523
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3866851, 1.3866916
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552874, 1.5555444
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2416711, 2.2417717
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5118858, 1.5121450
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5234780, 1.5235882
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1587563, 2.1590259
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949397, 0.8949407
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962954, 1.1963048

Time for backsubstitution: 4.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2644

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788271, upper bound: 0.5787363
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788271, upper bound: 0.5787350
time: 8.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311533, 1.4311528
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949554, 0.8949484
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3866856, 1.3866909
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5552876, 1.5555441
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2417111, 2.2417316
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5118848, 1.5121460
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5234773, 1.5235889
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1588030, 2.1589794
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949398, 0.8949407
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962954, 1.1963050

Time for backsubstitution: 4.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 829

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787023, upper bound: 0.5787936
time: 12.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787194, upper bound: 0.5787801
time: 18.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311670, 1.4311653
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949559, 0.8949301
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3856622, 1.3857701
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5441883, 1.5451888
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2281263, 2.2290602
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5001605, 1.5012150
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5169435, 1.5172641
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1501942, 2.1509123
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948695, 0.8948670
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962564, 1.1962656

Time for backsubstitution: 4.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2606

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786684, upper bound: 0.5785887
time: 39.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786871, upper bound: 0.5785604
time: 123.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311675, 1.4311649
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8949333, 0.8949529
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3857642, 1.3856680
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5449319, 1.5444450
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2289999, 2.2281871
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5009549, 1.5004208
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5171533, 1.5170546
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1506891, 2.1504173
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948659, 0.8948706
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962559, 1.1962661

Time for backsubstitution: 4.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2288

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777728, upper bound: 0.5778496
time: 33.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777588, upper bound: 0.5778650
time: 14.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311363, 1.4311275
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948466, 0.8948423
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3864419, 1.3864877
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5539502, 1.5542507
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2392249, 2.2401590
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5104852, 1.5107969
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5225776, 1.5226895
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1573746, 2.1579976
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949130, 0.8949138
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962538, 1.1962624

Time for backsubstitution: 4.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2047

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2643

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776649, upper bound: 0.5776298
time: 96.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776630, upper bound: 0.5776341
time: 197.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311360, 1.4311279
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948464, 0.8948424
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3864429, 1.3864865
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5539573, 1.5542439
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2393155, 2.2400684
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5104907, 1.5107915
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5225800, 1.5226871
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1573846, 2.1579876
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949128, 0.8949139
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962538, 1.1962624

Time for backsubstitution: 4.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3201

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5781842, upper bound: 0.5781849
time: 19.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5782000, upper bound: 0.5781695
time: 8.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311386, 1.4311279
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948445, 0.8948444
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3864574, 1.3864994
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5540134, 1.5543060
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2395926, 2.2403896
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5105722, 1.5108743
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5226141, 1.5227149
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1574798, 2.1580951
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949141, 0.8949152
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962531, 1.1962639

Time for backsubstitution: 4.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2618

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2142

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783448, upper bound: 0.5782906
time: 118.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783374, upper bound: 0.5782948
time: 16.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311386, 1.4311279
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8948444, 0.8948444
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3864574, 1.3864994
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5540131, 1.5543061
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2395926, 2.2403893
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5105722, 1.5108743
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5226148, 1.5227141
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1574798, 2.1580954
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949141, 0.8949152
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1962533, 1.1962637

Time for backsubstitution: 4.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2261

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783386, upper bound: 0.5782681
time: 37.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5782731, upper bound: 0.5783386
time: 49.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311428, 1.4311484
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8938588, 0.8940965
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3864883, 1.3864486
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5540330, 1.5542529
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2404497, 2.2397294
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5106099, 1.5108228
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5226068, 1.5227172
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1577072, 2.1576097
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948563, 0.8948573
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1958542, 1.1960808

Time for backsubstitution: 4.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2513

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776136, upper bound: 0.5778243
time: 36.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5776136, upper bound: 0.5778236
time: 49.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311426, 1.4311486
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8941005, 0.8938547
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3864795, 1.3864574
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5540321, 1.5542536
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2404277, 2.2397509
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5106058, 1.5108268
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5225976, 1.5227265
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1577778, 2.1575394
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8948565, 0.8948571
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1960716, 1.1958636

Time for backsubstitution: 4.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2563

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5781699, upper bound: 0.5781527
time: 8.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5781654, upper bound: 0.5781592
time: 194.34 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 207.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5787607, upper bound: 0.5785982
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5787582, upper bound: 0.5785988
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5785646, upper bound: 0.5786429
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5784863, upper bound: 0.5787221
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5786824, upper bound: 0.5778866
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5778753, upper bound: 0.5786958
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5786667, upper bound: 0.5786463
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5786853, upper bound: 0.5786261
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5788271, upper bound: 0.5787363
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5788271, upper bound: 0.5787350
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5787023, upper bound: 0.5787936
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5787194, upper bound: 0.5787801
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5786684, upper bound: 0.5785887
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5786871, upper bound: 0.5785604
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5777728, upper bound: 0.5778496
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5777588, upper bound: 0.5778650
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5776649, upper bound: 0.5776298
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5776630, upper bound: 0.5776341
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5781842, upper bound: 0.5781849
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5782000, upper bound: 0.5781695
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5783448, upper bound: 0.5782906
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5783374, upper bound: 0.5782948
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5783386, upper bound: 0.5782681
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5782731, upper bound: 0.5783386
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5776136, upper bound: 0.5778243
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5776136, upper bound: 0.5778236
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5781699, upper bound: 0.5781527
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 207.36
Output dim: 1, lower bound: -0.5781654, upper bound: 0.5781592
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5779919, upper bound: 0.5779181
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5779919, upper bound: 0.5779181
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5785487, upper bound: 0.5784470
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5785394, upper bound: 0.5784553
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5780713, upper bound: 0.5783140
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5782089, upper bound: 0.5781714
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5785713, upper bound: 0.5787588
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5786042, upper bound: 0.5787273
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5787630, upper bound: 0.5786212
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5787630, upper bound: 0.5786212
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5787966, upper bound: 0.5787120
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5787933, upper bound: 0.5787126
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5788239, upper bound: 0.5787385
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5788238, upper bound: 0.5787374
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5784730, upper bound: 0.5785217
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5784226, upper bound: 0.5785696
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5784593, upper bound: 0.5785255
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 207.36
Output dim: 1, lower bound: -0.5784310, upper bound: 0.5785558
Binary search (step 0): status=Status.UNKNOWN, k_low=2, k_high=8, k_mid=5, eps_mid=0.0195312, abs_max=0.8949273824691772
rel_dist={1: [-0.579017517234431, 0.5790172832286323]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2443

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232737, upper bound: 0.5232781
time: 56.11 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232737, upper bound: 0.5232781
time: 55.93 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 112.05 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 112.05
Output dim: 1, lower bound: -0.5232737, upper bound: 0.5232781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 112.05
Output dim: 1, lower bound: -0.5232737, upper bound: 0.5232781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185553, 1.4185553
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8648512, 0.8648511
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3593538, 1.3593543
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4950421, 1.4950415
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1963558, 2.1963644
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4509699, 1.4509715
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4446685, 1.4446869
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1161487, 2.1161671
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900347, 0.8900346
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1714802, 1.1714801

Time for backsubstitution: 4.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2049

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3341

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232202, upper bound: 0.5232742
time: 27.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232762, upper bound: 0.5232221
time: 57.13 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185553, 1.4185555
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8648517, 0.8648511
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3593543, 1.3593552
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4950416, 1.4950421
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1963716, 2.1963558
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4509709, 1.4509699
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4446738, 1.4446688
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1161544, 2.1161485
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900347, 0.8900346
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1714805, 1.1714802

Time for backsubstitution: 4.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3379

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 505

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232357, upper bound: 0.5232327
time: 53.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232319, upper bound: 0.5232396
time: 37.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 94.49 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 94.49
Output dim: 1, lower bound: -0.5232202, upper bound: 0.5232742
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 94.49
Output dim: 1, lower bound: -0.5232762, upper bound: 0.5232221
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 94.49
Output dim: 1, lower bound: -0.5232357, upper bound: 0.5232327
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 94.49
Output dim: 1, lower bound: -0.5232319, upper bound: 0.5232396

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4180982, 1.4180841
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8647399, 0.8647465
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3589139, 1.3589061
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4942182, 1.4941930
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1961963, 2.1961942
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4503102, 1.4502702
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4437633, 1.4437283
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1161470, 2.1161656
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900274, 0.8900273
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1710793, 1.1710992

Time for backsubstitution: 4.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2339

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231898, upper bound: 0.5232421
time: 38.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231831, upper bound: 0.5232475
time: 737.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4180841, 1.4180983
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8647466, 0.8647399
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3589056, 1.3589145
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4941934, 1.4942176
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1961854, 2.1962047
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4502687, 1.4503119
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4437103, 1.4437814
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1161470, 2.1161659
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900273, 0.8900274
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1710993, 1.1710792

Time for backsubstitution: 4.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232421, upper bound: 0.5230956
time: 94.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231517, upper bound: 0.5231914
time: 216.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185522, 1.4185524
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8648369, 0.8648351
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3593445, 1.3593454
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4950353, 1.4950358
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1963372, 2.1963234
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4509693, 1.4509683
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4446590, 1.4446564
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1161480, 2.1161423
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900343, 0.8900344
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1714721, 1.1714722

Time for backsubstitution: 4.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 297

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3367

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232342, upper bound: 0.5232321
time: 252.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232342, upper bound: 0.5232286
time: 227.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185522, 1.4185522
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8648357, 0.8648363
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3593447, 1.3593452
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4950351, 1.4950360
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1963391, 2.1963220
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4509693, 1.4509683
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4446614, 1.4446537
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1161485, 2.1161420
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900343, 0.8900343
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1714725, 1.1714720

Time for backsubstitution: 4.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2464

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2389

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231953, upper bound: 0.5231202
time: 18.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231170, upper bound: 0.5232061
time: 27.43 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 49.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 49.89
Output dim: 1, lower bound: -0.5231898, upper bound: 0.5232421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 49.89
Output dim: 1, lower bound: -0.5231831, upper bound: 0.5232475
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 49.89
Output dim: 1, lower bound: -0.5232421, upper bound: 0.5230956
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 49.89
Output dim: 1, lower bound: -0.5231517, upper bound: 0.5231914
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 49.89
Output dim: 1, lower bound: -0.5232342, upper bound: 0.5232321
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 49.89
Output dim: 1, lower bound: -0.5232342, upper bound: 0.5232286
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 49.89
Output dim: 1, lower bound: -0.5231953, upper bound: 0.5231202
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 49.89
Output dim: 1, lower bound: -0.5231170, upper bound: 0.5232061

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4180834, 1.4180670
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8647132, 0.8647194
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3587890, 1.3587914
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4916624, 1.4918730
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1955664, 2.1956162
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4476883, 1.4478912
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4426315, 1.4427016
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1142306, 2.1144509
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900212, 0.8900216
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1710496, 1.1710749

Time for backsubstitution: 4.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2439

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2981

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5228948, upper bound: 0.5231264
time: 102.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5230767, upper bound: 0.5229435
time: 16.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4180813, 1.4180691
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8647127, 0.8647199
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3587992, 1.3587811
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4918979, 1.4916372
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1956184, 2.1955643
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4479311, 1.4476485
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4427364, 1.4425967
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1144328, 2.1142492
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900217, 0.8900212
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1710548, 1.1710696

Time for backsubstitution: 4.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2640

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231008, upper bound: 0.5231611
time: 32.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5230969, upper bound: 0.5231692
time: 16.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4179250, 1.4179351
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8640676, 0.8640310
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3578148, 1.3578594
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4786738, 1.4792124
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1927776, 2.1929097
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4345696, 1.4351377
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4304380, 1.4309430
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1037433, 2.1041818
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8898701, 0.8898740
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1707239, 1.1707456

Time for backsubstitution: 4.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3519

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2445

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231413, upper bound: 0.5230084
time: 34.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231543, upper bound: 0.5229950
time: 59.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4179209, 1.4179392
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8640378, 0.8640608
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3578506, 1.3578236
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4791883, 1.4786978
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1928902, 2.1927972
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4350946, 1.4346129
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4308720, 1.4305092
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1041629, 2.1037621
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8898739, 0.8898702
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1707656, 1.1707039

Time for backsubstitution: 4.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2065

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2145

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5230179, upper bound: 0.5231771
time: 18.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231420, upper bound: 0.5230511
time: 59.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185522, 1.4185524
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8648369, 0.8648351
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3593445, 1.3593454
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4950353, 1.4950358
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1963372, 2.1963234
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4509693, 1.4509683
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4446590, 1.4446564
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1161480, 2.1161423
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900343, 0.8900344
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1714721, 1.1714722

Time for backsubstitution: 4.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3071

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 741

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232075, upper bound: 0.5231970
time: 121.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231994, upper bound: 0.5232010
time: 162.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185522, 1.4185524
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8648369, 0.8648351
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3593445, 1.3593454
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4950353, 1.4950358
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1963372, 2.1963234
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4509693, 1.4509683
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4446590, 1.4446564
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1161480, 2.1161423
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900343, 0.8900344
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1714721, 1.1714722

Time for backsubstitution: 4.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 750

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2376

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231968, upper bound: 0.5231551
time: 107.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231589, upper bound: 0.5231952
time: 106.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4184513, 1.4184500
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8644991, 0.8644840
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3590109, 1.3590286
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4803783, 1.4810750
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1920867, 2.1922219
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4358782, 1.4365313
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4371331, 1.4374053
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1055951, 2.1059675
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8899314, 0.8899323
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1712384, 1.1712400

Time for backsubstitution: 4.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 811

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231782, upper bound: 0.5231150
time: 43.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231863, upper bound: 0.5231069
time: 209.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4184499, 1.4184513
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8644831, 0.8644999
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3590281, 1.3590114
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4810742, 1.4803792
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1922383, 2.1920705
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4365324, 1.4358771
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4374130, 1.4371254
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1059737, 2.1055889
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8899324, 0.8899314
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1712403, 1.1712381

Time for backsubstitution: 4.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2065

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3201

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231025, upper bound: 0.5232009
time: 34.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231145, upper bound: 0.5231897
time: 26.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 65.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5228948, upper bound: 0.5231264
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5230767, upper bound: 0.5229435
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5231008, upper bound: 0.5231611
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5230969, upper bound: 0.5231692
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5231413, upper bound: 0.5230084
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5231543, upper bound: 0.5229950
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5230179, upper bound: 0.5231771
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5231420, upper bound: 0.5230511
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5232075, upper bound: 0.5231970
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5231994, upper bound: 0.5232010
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5231968, upper bound: 0.5231551
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5231589, upper bound: 0.5231952
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5231782, upper bound: 0.5231150
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5231863, upper bound: 0.5231069
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5231025, upper bound: 0.5232009
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 65.04
Output dim: 1, lower bound: -0.5231145, upper bound: 0.5231897

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4172153, 1.4171830
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8553685, 0.8555577
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3587615, 1.3587675
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4909275, 1.4911516
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1923199, 2.1922951
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4470392, 1.4472532
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4400362, 1.4401656
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1140151, 2.1142359
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900896, 0.8900900
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1676342, 1.1677445

Time for backsubstitution: 4.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 490

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2615

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5226985, upper bound: 0.5229535
time: 213.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5227199, upper bound: 0.5226956
time: 827.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 1045.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 1045.37
Output dim: 1, lower bound: -0.5226985, upper bound: 0.5229535
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 1045.37
Output dim: 1, lower bound: -0.5227199, upper bound: 0.5226956
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5230767, upper bound: 0.5229435
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5231008, upper bound: 0.5231611
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5230969, upper bound: 0.5231692
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5231413, upper bound: 0.5230084
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5231543, upper bound: 0.5229950
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5230179, upper bound: 0.5231771
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5231420, upper bound: 0.5230511
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5232075, upper bound: 0.5231970
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5231994, upper bound: 0.5232010
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5231968, upper bound: 0.5231551
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5231589, upper bound: 0.5231952
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5231782, upper bound: 0.5231150
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5231863, upper bound: 0.5231069
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5231025, upper bound: 0.5232009
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1045.37
Output dim: 1, lower bound: -0.5231145, upper bound: 0.5231897
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=4, k_mid=3, eps_mid=0.0117188, abs_max=0.8648518323898315
rel_dist={1: [-0.5234064232429227, 0.5234093100156474]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2167

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4946187, upper bound: 0.4946181
time: 151.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4946200, upper bound: 0.4946217
time: 16.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 168.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 168.38
Output dim: 1, lower bound: -0.4946187, upper bound: 0.4946181
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 168.38
Output dim: 1, lower bound: -0.4946200, upper bound: 0.4946217

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122179, 1.4122180
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497727, 0.8497566
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456385, 1.3456385
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4628673, 1.4630926
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1725678, 2.1727684
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4182644, 1.4185088
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4027021, 1.4027462
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0930946, 2.0932157
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875470, 0.8875468
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589421, 1.1589845

Time for backsubstitution: 4.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2603

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2454

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945760, upper bound: 0.4945647
time: 40.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945642, upper bound: 0.4945804
time: 45.98 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122181, 1.4122179
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497565, 0.8497728
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456388, 1.3456385
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4630926, 1.4628673
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1727681, 2.1725678
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4185085, 1.4182644
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4027462, 1.4027021
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0932157, 2.0930948
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875468, 0.8875471
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589845, 1.1589420

Time for backsubstitution: 4.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2947

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945250, upper bound: 0.4945292
time: 86.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945250, upper bound: 0.4945272
time: 100.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 191.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 191.55
Output dim: 1, lower bound: -0.4945760, upper bound: 0.4945647
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 191.55
Output dim: 1, lower bound: -0.4945642, upper bound: 0.4945804
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 191.55
Output dim: 1, lower bound: -0.4945250, upper bound: 0.4945292
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 191.55
Output dim: 1, lower bound: -0.4945250, upper bound: 0.4945272

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122176, 1.4122179
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497722, 0.8497559
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456311, 1.3456326
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4628363, 1.4630752
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1723666, 2.1726341
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4182248, 1.4184779
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4026833, 1.4027302
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0930495, 2.0931835
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875465, 0.8875463
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589417, 1.1589842

Time for backsubstitution: 4.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3084

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945719, upper bound: 0.4945669
time: 31.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945719, upper bound: 0.4945663
time: 16.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122176, 1.4122179
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497722, 0.8497561
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456326, 1.3456314
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4628496, 1.4630618
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1724334, 2.1725667
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4182334, 1.4184692
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4026864, 1.4027274
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0930626, 2.0931706
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875465, 0.8875463
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589417, 1.1589842

Time for backsubstitution: 4.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2079

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945253, upper bound: 0.4945589
time: 21.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945423, upper bound: 0.4945418
time: 101.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122175, 1.4122169
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497527, 0.8497654
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456383, 1.3456385
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4630921, 1.4628659
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1727633, 2.1725636
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4185083, 1.4182633
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4027441, 1.4027014
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0932152, 2.0930943
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875468, 0.8875471
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589830, 1.1589406

Time for backsubstitution: 4.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2047

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2280

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4941894, upper bound: 0.4942482
time: 49.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4942396, upper bound: 0.4941927
time: 1073.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122173, 1.4122179
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497565, 0.8497689
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456388, 1.3456384
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4630926, 1.4628667
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1727681, 2.1725628
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4185085, 1.4182639
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4027462, 1.4026999
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0932152, 2.0930948
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875468, 0.8875470
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589845, 1.1589403

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 3098

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 831

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945078, upper bound: 0.4945130
time: 114.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945114, upper bound: 0.4945117
time: 15.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 134.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 134.03
Output dim: 1, lower bound: -0.4945719, upper bound: 0.4945669
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 134.03
Output dim: 1, lower bound: -0.4945719, upper bound: 0.4945663
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 134.03
Output dim: 1, lower bound: -0.4945253, upper bound: 0.4945589
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 134.03
Output dim: 1, lower bound: -0.4945423, upper bound: 0.4945418
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 134.03
Output dim: 1, lower bound: -0.4941894, upper bound: 0.4942482
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 134.03
Output dim: 1, lower bound: -0.4942396, upper bound: 0.4941927
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 134.03
Output dim: 1, lower bound: -0.4945078, upper bound: 0.4945130
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 134.03
Output dim: 1, lower bound: -0.4945114, upper bound: 0.4945117

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122176, 1.4122179
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497722, 0.8497559
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456311, 1.3456326
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4628363, 1.4630752
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1723666, 2.1726341
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4182248, 1.4184779
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4026833, 1.4027302
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0930495, 2.0931835
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875465, 0.8875463
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589417, 1.1589842

Time for backsubstitution: 4.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2280

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945186, upper bound: 0.4944962
time: 21.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945089, upper bound: 0.4945007
time: 20.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122176, 1.4122179
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497722, 0.8497559
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456311, 1.3456326
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4628363, 1.4630752
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1723666, 2.1726341
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4182248, 1.4184779
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4026833, 1.4027302
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0930495, 2.0931835
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875465, 0.8875463
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589417, 1.1589842

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2981

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2519

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945781, upper bound: 0.4945655
time: 52.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945781, upper bound: 0.4945672
time: 40.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4121542, 1.4121525
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8487824, 0.8487948
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3455197, 1.3455148
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4629009, 1.4631144
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1715126, 2.1716306
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4181681, 1.4184051
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4016826, 1.4017351
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0931139, 2.0932231
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875526, 0.8875524
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1572880, 1.1573906

Time for backsubstitution: 4.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2301

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4943126, upper bound: 0.4943395
time: 147.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4943083, upper bound: 0.4943432
time: 202.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4121523, 1.4121542
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8488110, 0.8487662
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3455161, 1.3455184
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4629023, 1.4631128
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1714978, 2.1716454
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4181693, 1.4184039
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4016941, 1.4017237
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0931153, 2.0932219
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875528, 0.8875521
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1573480, 1.1573305

Time for backsubstitution: 4.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2373

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2449

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944938, upper bound: 0.4944934
time: 33.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944884, upper bound: 0.4944941
time: 31.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122090, 1.4122107
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497436, 0.8497558
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456192, 1.3456178
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4630578, 1.4628154
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1726298, 2.1724858
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4184458, 1.4182235
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4024243, 1.4023541
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0928457, 2.0927460
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875450, 0.8875452
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589835, 1.1589392

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945058, upper bound: 0.4944811
time: 16.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944780, upper bound: 0.4945141
time: 23.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122100, 1.4122097
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497431, 0.8497562
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456178, 1.3456190
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4630628, 1.4628317
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1726909, 2.1724248
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4184749, 1.4182010
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4024200, 1.4023783
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0928667, 2.0927253
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875450, 0.8875452
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589835, 1.1589392

Time for backsubstitution: 4.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2983

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4941828, upper bound: 0.4941857
time: 162.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4941881, upper bound: 0.4941797
time: 127.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 294.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4945186, upper bound: 0.4944962
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4945089, upper bound: 0.4945007
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4945781, upper bound: 0.4945655
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4945781, upper bound: 0.4945672
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4943126, upper bound: 0.4943395
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4943083, upper bound: 0.4943432
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4944938, upper bound: 0.4944934
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4944884, upper bound: 0.4944941
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4945058, upper bound: 0.4944811
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4944780, upper bound: 0.4945141
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4941828, upper bound: 0.4941857
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 294.71
Output dim: 1, lower bound: -0.4941881, upper bound: 0.4941797

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122105, 1.4122107
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8494184, 0.8494071
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3455877, 1.3455884
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4627934, 1.4630330
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1723342, 2.1726031
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4182231, 1.4184760
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4026768, 1.4027238
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0929260, 2.0930572
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875242, 0.8875247
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1586108, 1.1586524

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2449

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2202

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944985, upper bound: 0.4944803
time: 15.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944983, upper bound: 0.4944777
time: 158.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122105, 1.4122107
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8494234, 0.8494021
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3455870, 1.3455889
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4627943, 1.4630320
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1723351, 2.1726024
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4182231, 1.4184760
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4026768, 1.4027238
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0929229, 2.0930600
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875250, 0.8875241
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1586101, 1.1586534

Time for backsubstitution: 4.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2379

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2304

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4937865, upper bound: 0.4937782
time: 281.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.4937865, upper bound: 0.4937817
time: 294.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122176, 1.4122179
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497722, 0.8497559
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456311, 1.3456326
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4628363, 1.4630752
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1723666, 2.1726341
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4182248, 1.4184779
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4026833, 1.4027302
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0930495, 2.0931835
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875465, 0.8875463
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589417, 1.1589842

Time for backsubstitution: 4.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2424

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945447, upper bound: 0.4944903
time: 21.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945040, upper bound: 0.4945353
time: 183.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122176, 1.4122179
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8497722, 0.8497559
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3456311, 1.3456326
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4628363, 1.4630752
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1723666, 2.1726341
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4182248, 1.4184779
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4026833, 1.4027302
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0930495, 2.0931835
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875465, 0.8875463
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1589417, 1.1589842

Time for backsubstitution: 4.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2050

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2300

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945300, upper bound: 0.4945161
time: 25.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945344, upper bound: 0.4945129
time: 91.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4121480, 1.4121499
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8486652, 0.8486271
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3452448, 1.3452518
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4617808, 1.4620117
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1699309, 2.1701016
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4165215, 1.4167787
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4009655, 1.4010062
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0919724, 2.0921528
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875109, 0.8875101
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1573439, 1.1573268

Time for backsubstitution: 4.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3589
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 2445

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 762

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944914, upper bound: 0.4944917
time: 61.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944959, upper bound: 0.4944890
time: 210.70 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 276.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 276.84
Output dim: 1, lower bound: -0.4944985, upper bound: 0.4944803
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 276.84
Output dim: 1, lower bound: -0.4944983, upper bound: 0.4944777
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 276.84
Output dim: 1, lower bound: -0.4937865, upper bound: 0.4937782
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 276.84
Output dim: 1, lower bound: -0.4937865, upper bound: 0.4937817
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 276.84
Output dim: 1, lower bound: -0.4945447, upper bound: 0.4944903
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 276.84
Output dim: 1, lower bound: -0.4945040, upper bound: 0.4945353
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 276.84
Output dim: 1, lower bound: -0.4945300, upper bound: 0.4945161
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 276.84
Output dim: 1, lower bound: -0.4945344, upper bound: 0.4945129
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 276.84
Output dim: 1, lower bound: -0.4944914, upper bound: 0.4944917
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 276.84
Output dim: 1, lower bound: -0.4944959, upper bound: 0.4944890
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 276.84
Output dim: 1, lower bound: -0.4944884, upper bound: 0.4944941
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 276.84
Output dim: 1, lower bound: -0.4945058, upper bound: 0.4944811
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 276.84
Output dim: 1, lower bound: -0.4944780, upper bound: 0.4945141
Binary search (step 2): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.8498140573501587
rel_dist={1: [-0.49469755591069386, 0.49469970537319075]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 12557.61 seconds
