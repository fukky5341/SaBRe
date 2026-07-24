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
execution time: IAR + LP analysis = 5.54 + 120.63 = 126.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.6552173, upper bound: 0.6552176


# Binary Search by BASE starts (time budget: 17873.84 seconds, max iter: 100)

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
Binary search time: 209.55 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Relational Split (RS_dual_Z) starts
Time budget: 17664.29 seconds

## Binary search (step 0) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3115

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5789650, upper bound: 0.5788832
time: 12.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788871, upper bound: 0.5789646
time: 49.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 62.30 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 62.30
Output dim: 1, lower bound: -0.5789650, upper bound: 0.5788832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 62.30
Output dim: 1, lower bound: -0.5788871, upper bound: 0.5789646

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311681, 1.4311696
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8951150, 0.8951170
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3869525, 1.3869450
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5565319, 1.5565219
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2383823, 2.2384534
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5129905, 1.5129886
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5233393, 1.5233710
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1591656, 2.1591730
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949445, 0.8949444
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963333, 1.1963336

Time for backsubstitution: 4.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 3113

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5789099, upper bound: 0.5787577
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788346, upper bound: 0.5788297
time: 7.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311697, 1.4311681
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8951169, 0.8951150
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3869449, 1.3869528
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5565219, 1.5565321
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2384534, 2.2383823
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5129886, 1.5129901
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5233710, 1.5233393
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1591733, 2.1591659
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949444, 0.8949445
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963338, 1.1963332

Time for backsubstitution: 4.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3113

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5788332, upper bound: 0.5788326
time: 65.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5787588, upper bound: 0.5789076
time: 58.18 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 127.61 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 127.61
Output dim: 1, lower bound: -0.5789099, upper bound: 0.5787577
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 127.61
Output dim: 1, lower bound: -0.5788346, upper bound: 0.5788297
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 127.61
Output dim: 1, lower bound: -0.5788332, upper bound: 0.5788326
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 127.61
Output dim: 1, lower bound: -0.5787588, upper bound: 0.5789076

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311874, 1.4311910
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950835, 0.8950854
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3866175, 1.3866127
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5548246, 1.5548077
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2280638, 2.2286139
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5107039, 1.5107176
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228853, 1.5229216
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1562045, 2.1562588
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949127, 0.8949095
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963257, 1.1963261

Time for backsubstitution: 4.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786575, upper bound: 0.5784471
time: 60.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785954, upper bound: 0.5785077
time: 10.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311893, 1.4311888
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950833, 0.8950855
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3866204, 1.3866096
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5548179, 1.5548143
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2285426, 2.2281353
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5107192, 1.5107024
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228896, 1.5229170
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1562512, 2.1562119
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949097, 0.8949124
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963257, 1.1963261

Time for backsubstitution: 4.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785788, upper bound: 0.5785236
time: 38.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785220, upper bound: 0.5785850
time: 44.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311888, 1.4311893
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950855, 0.8950834
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3866094, 1.3866205
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5548143, 1.5548179
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2281353, 2.2285426
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5107025, 1.5107193
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5229170, 1.5228899
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1562116, 2.1562514
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949124, 0.8949096
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963261, 1.1963257

Time for backsubstitution: 4.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785820, upper bound: 0.5785188
time: 351.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785236, upper bound: 0.5785833
time: 39.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311910, 1.4311874
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950852, 0.8950835
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3866128, 1.3866174
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5548077, 1.5548246
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2286141, 2.2280641
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5107177, 1.5107039
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5229216, 1.5228853
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1562588, 2.1562042
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949096, 0.8949126
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963261, 1.1963257

Time for backsubstitution: 4.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785057, upper bound: 0.5785978
time: 27.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784462, upper bound: 0.5786576
time: 18.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 50.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 50.40
Output dim: 1, lower bound: -0.5786575, upper bound: 0.5784471
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 50.40
Output dim: 1, lower bound: -0.5785954, upper bound: 0.5785077
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 50.40
Output dim: 1, lower bound: -0.5785788, upper bound: 0.5785236
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 50.40
Output dim: 1, lower bound: -0.5785220, upper bound: 0.5785850
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 50.40
Output dim: 1, lower bound: -0.5785820, upper bound: 0.5785188
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 50.40
Output dim: 1, lower bound: -0.5785236, upper bound: 0.5785833
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 50.40
Output dim: 1, lower bound: -0.5785057, upper bound: 0.5785978
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 50.40
Output dim: 1, lower bound: -0.5784462, upper bound: 0.5786576

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311812, 1.4311852
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950768, 0.8950784
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865567, 1.3865569
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545216, 1.5545343
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2264607, 2.2272408
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103372, 1.5103892
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227740, 1.5228209
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1557758, 2.1558633
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949105, 0.8949058
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963239, 1.1963242

Time for backsubstitution: 4.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786528, upper bound: 0.5783998
time: 22.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5786095, upper bound: 0.5784433
time: 11.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311817, 1.4311846
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950766, 0.8950785
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865615, 1.3865519
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545516, 1.5545048
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2266924, 2.2270103
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103754, 1.5103509
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227849, 1.5228101
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558089, 2.1558304
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949089, 0.8949074
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963239, 1.1963242

Time for backsubstitution: 4.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785935, upper bound: 0.5784553
time: 215.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785508, upper bound: 0.5785037
time: 46.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311831, 1.4311831
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950765, 0.8950786
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865596, 1.3865539
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545149, 1.5545412
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2269390, 2.2267637
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103525, 1.5103736
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227783, 1.5228165
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558228, 2.1558161
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949075, 0.8949087
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963239, 1.1963242

Time for backsubstitution: 4.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785776, upper bound: 0.5784763
time: 55.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785352, upper bound: 0.5785161
time: 55.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311836, 1.4311826
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950765, 0.8950787
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865649, 1.3865489
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545444, 1.5545113
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2271693, 2.2265317
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103911, 1.5103356
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227892, 1.5228057
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558561, 2.1557832
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949060, 0.8949102
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963239, 1.1963243

Time for backsubstitution: 4.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785165, upper bound: 0.5785376
time: 38.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784759, upper bound: 0.5785734
time: 16.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311826, 1.4311836
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950787, 0.8950765
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865489, 1.3865647
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545115, 1.5545444
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2265317, 2.2271690
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103353, 1.5103910
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228055, 1.5227892
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1557832, 2.1558561
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949103, 0.8949059
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963243, 1.1963238

Time for backsubstitution: 4.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785780, upper bound: 0.5784726
time: 27.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785344, upper bound: 0.5785164
time: 117.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311831, 1.4311831
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950787, 0.8950766
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865539, 1.3865597
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545411, 1.5545149
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2267640, 2.2269390
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103735, 1.5103526
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228164, 1.5227784
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558161, 2.1558228
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949087, 0.8949075
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963243, 1.1963238

Time for backsubstitution: 4.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785179, upper bound: 0.5785357
time: 18.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784760, upper bound: 0.5785746
time: 17.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311848, 1.4311817
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950785, 0.8950766
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865520, 1.3865616
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545049, 1.5545514
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2270100, 2.2266922
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103511, 1.5103754
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228102, 1.5227848
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558304, 2.1558089
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949074, 0.8949089
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963243, 1.1963238

Time for backsubstitution: 4.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785005, upper bound: 0.5785535
time: 39.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784591, upper bound: 0.5785941
time: 15.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311852, 1.4311811
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950784, 0.8950768
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865570, 1.3865566
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545344, 1.5545216
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2272408, 2.2264605
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103892, 1.5103372
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228207, 1.5227739
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558633, 2.1557758
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949058, 0.8949105
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963243, 1.1963239

Time for backsubstitution: 4.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784393, upper bound: 0.5786099
time: 292.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784000, upper bound: 0.5786533
time: 45.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 342.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5786528, upper bound: 0.5783998
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5786095, upper bound: 0.5784433
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5785935, upper bound: 0.5784553
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5785508, upper bound: 0.5785037
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5785776, upper bound: 0.5784763
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5785352, upper bound: 0.5785161
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5785165, upper bound: 0.5785376
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5784759, upper bound: 0.5785734
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5785780, upper bound: 0.5784726
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5785344, upper bound: 0.5785164
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5785179, upper bound: 0.5785357
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5784760, upper bound: 0.5785746
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5785005, upper bound: 0.5785535
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5784591, upper bound: 0.5785941
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5784393, upper bound: 0.5786099
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 342.56
Output dim: 1, lower bound: -0.5784000, upper bound: 0.5786533

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311814, 1.4311852
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950729, 0.8950744
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865572, 1.3865576
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545267, 1.5545394
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2264638, 2.2272439
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103431, 1.5103953
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227616, 1.5228093
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1557770, 2.1558642
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949105, 0.8949058
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963276, 1.1963285

Time for backsubstitution: 4.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785966, upper bound: 0.5783163
time: 75.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785367, upper bound: 0.5783367
time: 25.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311814, 1.4311852
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950729, 0.8950744
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865575, 1.3865575
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545264, 1.5545396
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2264638, 2.2272439
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103431, 1.5103953
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227623, 1.5228086
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1557770, 2.1558642
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949105, 0.8949058
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963283, 1.1963279

Time for backsubstitution: 4.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785544, upper bound: 0.5783568
time: 41.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784950, upper bound: 0.5783703
time: 8.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311819, 1.4311848
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950726, 0.8950744
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865622, 1.3865526
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545565, 1.5545099
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2266955, 2.2270133
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103812, 1.5103570
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227723, 1.5227984
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558099, 2.1558313
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949089, 0.8949074
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963276, 1.1963285

Time for backsubstitution: 4.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785238, upper bound: 0.5783448
time: 34.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785079, upper bound: 0.5784025
time: 52.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311819, 1.4311848
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950726, 0.8950745
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865622, 1.3865525
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545565, 1.5545100
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2266955, 2.2270133
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103812, 1.5103571
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227733, 1.5227977
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558099, 2.1558313
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949089, 0.8949074
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963280, 1.1963280

Time for backsubstitution: 4.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784828, upper bound: 0.5783843
time: 17.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784655, upper bound: 0.5784458
time: 49.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311833, 1.4311833
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950726, 0.8950745
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865603, 1.3865545
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545200, 1.5545464
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2269421, 2.2267671
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103588, 1.5103798
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227659, 1.5228049
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558242, 2.1558170
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949075, 0.8949087
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963276, 1.1963285

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785190, upper bound: 0.5783885
time: 88.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784612, upper bound: 0.5784101
time: 89.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311833, 1.4311833
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950726, 0.8950745
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865603, 1.3865545
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545200, 1.5545465
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2269421, 2.2267671
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103588, 1.5103798
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227666, 1.5228041
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558242, 2.1558173
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949075, 0.8949087
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963280, 1.1963280

Time for backsubstitution: 4.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784793, upper bound: 0.5784337
time: 35.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784180, upper bound: 0.5784512
time: 9.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311838, 1.4311827
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950724, 0.8950747
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865653, 1.3865495
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545498, 1.5545166
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2271724, 2.2265348
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103970, 1.5103416
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227768, 1.5227940
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558571, 2.1557844
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949060, 0.8949102
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963273, 1.1963286

Time for backsubstitution: 4.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784487, upper bound: 0.5784224
time: 29.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784300, upper bound: 0.5784806
time: 17.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311838, 1.4311827
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950724, 0.8950747
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865653, 1.3865495
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545496, 1.5545166
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2271724, 2.2265348
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103970, 1.5103416
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227776, 1.5227933
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558571, 2.1557844
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949060, 0.8949102
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963280, 1.1963280

Time for backsubstitution: 4.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784075, upper bound: 0.5784614
time: 101.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783890, upper bound: 0.5785220
time: 13.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311829, 1.4311838
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950748, 0.8950724
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865494, 1.3865653
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545164, 1.5545497
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2265348, 2.2271724
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103416, 1.5103970
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227933, 1.5227776
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1557841, 2.1558571
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949103, 0.8949059
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963280, 1.1963280

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785207, upper bound: 0.5783882
time: 122.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784613, upper bound: 0.5784096
time: 13.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311829, 1.4311838
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950748, 0.8950725
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865494, 1.3865652
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545164, 1.5545497
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2265348, 2.2271724
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103416, 1.5103971
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227940, 1.5227768
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1557841, 2.1558571
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949103, 0.8949059
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963288, 1.1963274

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784804, upper bound: 0.5784310
time: 33.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784186, upper bound: 0.5784512
time: 20.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311833, 1.4311832
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950745, 0.8950725
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865546, 1.3865603
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545465, 1.5545200
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2267671, 2.2269421
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103798, 1.5103586
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228043, 1.5227666
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558170, 2.1558242
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949087, 0.8949075
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963280, 1.1963280

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784508, upper bound: 0.5784185
time: 136.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784311, upper bound: 0.5784785
time: 89.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311833, 1.4311832
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950745, 0.8950725
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865546, 1.3865602
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545462, 1.5545201
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2267671, 2.2269421
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103798, 1.5103588
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228047, 1.5227659
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558170, 2.1558242
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949087, 0.8949075
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963285, 1.1963276

Time for backsubstitution: 4.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784101, upper bound: 0.5784616
time: 84.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783899, upper bound: 0.5785212
time: 77.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311848, 1.4311817
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950745, 0.8950725
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865525, 1.3865622
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545098, 1.5545566
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2270136, 2.2266953
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103569, 1.5103815
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227976, 1.5227731
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558313, 2.1558099
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949074, 0.8949089
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963280, 1.1963280

Time for backsubstitution: 4.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784449, upper bound: 0.5784627
time: 31.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783844, upper bound: 0.5784838
time: 27.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311848, 1.4311817
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950745, 0.8950726
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865527, 1.3865622
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545098, 1.5545566
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2270136, 2.2266953
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103569, 1.5103815
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5227985, 1.5227724
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558313, 2.1558099
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949074, 0.8949089
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963285, 1.1963276

Time for backsubstitution: 4.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784024, upper bound: 0.5785054
time: 32.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783432, upper bound: 0.5785242
time: 177.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311852, 1.4311812
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950745, 0.8950728
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865575, 1.3865572
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545396, 1.5545267
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2272439, 2.2264638
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103955, 1.5103433
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228086, 1.5227622
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558642, 2.1557770
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949058, 0.8949105
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963278, 1.1963282

Time for backsubstitution: 4.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783730, upper bound: 0.5784960
time: 42.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783564, upper bound: 0.5785550
time: 49.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311852, 1.4311812
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950743, 0.8950728
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865575, 1.3865572
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545393, 1.5545268
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2272439, 2.2264638
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5103951, 1.5103433
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228093, 1.5227615
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1558642, 2.1557770
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949058, 0.8949105
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963285, 1.1963276

Time for backsubstitution: 4.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783336, upper bound: 0.5785331
time: 9.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783124, upper bound: 0.5785966
time: 11.51 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5785966, upper bound: 0.5783163
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5785367, upper bound: 0.5783367
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5785544, upper bound: 0.5783568
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784950, upper bound: 0.5783703
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5785238, upper bound: 0.5783448
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5785079, upper bound: 0.5784025
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784828, upper bound: 0.5783843
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784655, upper bound: 0.5784458
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5785190, upper bound: 0.5783885
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784612, upper bound: 0.5784101
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784793, upper bound: 0.5784337
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784180, upper bound: 0.5784512
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784487, upper bound: 0.5784224
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784300, upper bound: 0.5784806
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784075, upper bound: 0.5784614
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5783890, upper bound: 0.5785220
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5785207, upper bound: 0.5783882
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784613, upper bound: 0.5784096
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784804, upper bound: 0.5784310
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784186, upper bound: 0.5784512
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784508, upper bound: 0.5784185
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784311, upper bound: 0.5784785
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784101, upper bound: 0.5784616
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5783899, upper bound: 0.5785212
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784449, upper bound: 0.5784627
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5783844, upper bound: 0.5784838
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5784024, upper bound: 0.5785054
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5783432, upper bound: 0.5785242
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5783730, upper bound: 0.5784960
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5783564, upper bound: 0.5785550
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5783336, upper bound: 0.5785331
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.55
Output dim: 1, lower bound: -0.5783124, upper bound: 0.5785966

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311757, 1.4311806
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950860, 0.8950870
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865350, 1.3865364
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545464, 1.5545548
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2250707, 2.2259331
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5100927, 1.5101526
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228413, 1.5229068
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1550879, 2.1551924
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949096, 0.8949047
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963092, 1.1963098

Time for backsubstitution: 4.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2438

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5785246, upper bound: 0.5782253
time: 27.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784899, upper bound: 0.5782347
time: 37.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311767, 1.4311799
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950854, 0.8950876
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865360, 1.3865347
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545421, 1.5545533
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2251084, 2.2258511
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5100951, 1.5101446
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228547, 1.5228890
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1550951, 2.1551752
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949094, 0.8949050
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963087, 1.1963100

Time for backsubstitution: 4.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2438

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784646, upper bound: 0.5782401
time: 110.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784300, upper bound: 0.5782384
time: 32.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4311757, 1.4311806
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8950860, 0.8950870
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3865350, 1.3865364
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.5545461, 1.5545549
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.2250707, 2.2259331
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.5100925, 1.5101526
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.5228418, 1.5229061
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1550879, 2.1551924
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8949096, 0.8949047
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1963097, 1.1963091

Time for backsubstitution: 4.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2438

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784819, upper bound: 0.5782713
time: 313.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5784493, upper bound: 0.5782785
time: 46.16 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 363.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 363.86
Output dim: 1, lower bound: -0.5785246, upper bound: 0.5782253
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 363.86
Output dim: 1, lower bound: -0.5784899, upper bound: 0.5782347
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 363.86
Output dim: 1, lower bound: -0.5784646, upper bound: 0.5782401
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 363.86
Output dim: 1, lower bound: -0.5784300, upper bound: 0.5782384
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 363.86
Output dim: 1, lower bound: -0.5784819, upper bound: 0.5782713
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 363.86
Output dim: 1, lower bound: -0.5784493, upper bound: 0.5782785
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784950, upper bound: 0.5783703
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5785238, upper bound: 0.5783448
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5785079, upper bound: 0.5784025
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784828, upper bound: 0.5783843
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784655, upper bound: 0.5784458
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5785190, upper bound: 0.5783885
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784612, upper bound: 0.5784101
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784793, upper bound: 0.5784337
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784180, upper bound: 0.5784512
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784487, upper bound: 0.5784224
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784300, upper bound: 0.5784806
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784075, upper bound: 0.5784614
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5783890, upper bound: 0.5785220
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5785207, upper bound: 0.5783882
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784613, upper bound: 0.5784096
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784804, upper bound: 0.5784310
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784186, upper bound: 0.5784512
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784508, upper bound: 0.5784185
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784311, upper bound: 0.5784785
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784101, upper bound: 0.5784616
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5783899, upper bound: 0.5785212
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784449, upper bound: 0.5784627
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5783844, upper bound: 0.5784838
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5784024, upper bound: 0.5785054
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5783432, upper bound: 0.5785242
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5783730, upper bound: 0.5784960
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5783564, upper bound: 0.5785550
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5783336, upper bound: 0.5785331
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 363.86
Output dim: 1, lower bound: -0.5783124, upper bound: 0.5785966
Binary search (step 0): status=Status.UNKNOWN, k_low=2, k_high=8, k_mid=5, eps_mid=0.0195312, abs_max=0.8949273824691772
rel_dist={1: [-0.579017517234431, 0.5790172832286323]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3115

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5233710, upper bound: 0.5233218
time: 580.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5233191, upper bound: 0.5233733
time: 33.83 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 614.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 614.29
Output dim: 1, lower bound: -0.5233710, upper bound: 0.5233218
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 614.29
Output dim: 1, lower bound: -0.5233191, upper bound: 0.5233733

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185545, 1.4185555
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8650395, 0.8650407
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3595701, 1.3595654
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4951224, 1.4951162
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1927967, 2.1928396
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4506636, 1.4506625
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4442365, 1.4442555
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1150343, 2.1150384
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900328, 0.8900326
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1715134, 1.1715136

Time for backsubstitution: 4.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3113

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5233368, upper bound: 0.5232391
time: 98.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232859, upper bound: 0.5232922
time: 60.90 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185555, 1.4185545
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8650407, 0.8650395
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3595654, 1.3595700
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4951162, 1.4951223
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1928396, 2.1927967
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4506626, 1.4506636
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4442556, 1.4442364
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1150386, 2.1150341
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8900326, 0.8900328
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1715137, 1.1715133

Time for backsubstitution: 4.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 3113

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232898, upper bound: 0.5232864
time: 45.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232371, upper bound: 0.5233411
time: 339.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 390.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 390.55
Output dim: 1, lower bound: -0.5233368, upper bound: 0.5232391
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 390.55
Output dim: 1, lower bound: -0.5232859, upper bound: 0.5232922
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 390.55
Output dim: 1, lower bound: -0.5232898, upper bound: 0.5232864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 390.55
Output dim: 1, lower bound: -0.5232371, upper bound: 0.5233411

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185746, 1.4185767
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8650080, 0.8650090
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3592347, 1.3592318
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4934149, 1.4934046
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1824787, 2.1828086
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4483775, 1.4483855
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4437826, 1.4438043
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1120727, 2.1121054
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8899996, 0.8899977
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1715058, 1.1715060

Time for backsubstitution: 4.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232021, upper bound: 0.5230649
time: 131.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231634, upper bound: 0.5231067
time: 28.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185760, 1.4185755
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8650078, 0.8650091
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3592366, 1.3592300
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4934108, 1.4934087
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1827657, 2.1825216
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4483865, 1.4483763
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4437852, 1.4438016
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1121008, 2.1120772
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8899979, 0.8899995
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1715058, 1.1715060

Time for backsubstitution: 4.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231525, upper bound: 0.5231205
time: 224.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231135, upper bound: 0.5231574
time: 29.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185755, 1.4185759
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8650090, 0.8650078
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3592299, 1.3592365
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4934087, 1.4934108
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1825211, 2.1827657
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4483765, 1.4483865
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4438016, 1.4437852
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1120770, 2.1121011
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8899996, 0.8899978
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1715060, 1.1715058

Time for backsubstitution: 4.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231533, upper bound: 0.5231146
time: 94.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231155, upper bound: 0.5231553
time: 33.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185767, 1.4185747
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8650090, 0.8650079
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3592318, 1.3592347
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4934049, 1.4934146
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1828082, 2.1824787
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4483856, 1.4483774
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4438043, 1.4437826
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1121051, 2.1120727
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8899977, 0.8899996
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1715058, 1.1715058

Time for backsubstitution: 4.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231040, upper bound: 0.5230662
time: 162.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5230622, upper bound: 0.5232055
time: 69.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 237.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 237.62
Output dim: 1, lower bound: -0.5232021, upper bound: 0.5230649
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 237.62
Output dim: 1, lower bound: -0.5231634, upper bound: 0.5231067
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 237.62
Output dim: 1, lower bound: -0.5231525, upper bound: 0.5231205
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 237.62
Output dim: 1, lower bound: -0.5231135, upper bound: 0.5231574
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 237.62
Output dim: 1, lower bound: -0.5231533, upper bound: 0.5231146
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 237.62
Output dim: 1, lower bound: -0.5231155, upper bound: 0.5231553
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 237.62
Output dim: 1, lower bound: -0.5231040, upper bound: 0.5230662
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 237.62
Output dim: 1, lower bound: -0.5230622, upper bound: 0.5232055

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185686, 1.4185710
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8650011, 0.8650021
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3591739, 1.3591741
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4931116, 1.4931194
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1808751, 2.1813433
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4480103, 1.4480417
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4436710, 1.4436994
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1116443, 2.1116967
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8899968, 0.8899940
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1715040, 1.1715041

Time for backsubstitution: 4.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5232032, upper bound: 0.5230261
time: 35.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231633, upper bound: 0.5230607
time: 207.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185691, 1.4185708
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8650010, 0.8650021
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3591770, 1.3591712
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4931297, 1.4931016
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1810143, 2.1812050
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4480332, 1.4480188
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4436777, 1.4436928
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1116638, 2.1116767
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8899959, 0.8899950
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1715040, 1.1715043

Time for backsubstitution: 4.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231633, upper bound: 0.5230237
time: 716.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231234, upper bound: 0.5231073
time: 17.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185698, 1.4185698
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8650010, 0.8650021
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3591758, 1.3591723
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4931078, 1.4931235
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1811621, 2.1810572
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4480199, 1.4480324
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4436738, 1.4436967
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1116724, 2.1116683
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8899951, 0.8899958
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1715040, 1.1715043

Time for backsubstitution: 4.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231495, upper bound: 0.5230776
time: 19.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231119, upper bound: 0.5231160
time: 456.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185703, 1.4185696
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8650010, 0.8650022
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3591787, 1.3591692
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4931254, 1.4931056
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1813004, 2.1809180
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4480428, 1.4480095
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4436805, 1.4436902
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1116924, 2.1116486
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8899941, 0.8899966
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1715040, 1.1715043

Time for backsubstitution: 4.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231114, upper bound: 0.5231178
time: 27.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5230726, upper bound: 0.5231574
time: 46.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4185696, 1.4185702
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8650023, 0.8650009
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3591691, 1.3591788
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4931059, 1.4931254
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1809180, 2.1813002
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4480094, 1.4480428
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4436901, 1.4436804
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.1116486, 2.1116924
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8899968, 0.8899941
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1715043, 1.1715039

Time for backsubstitution: 4.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231520, upper bound: 0.5230729
time: 303.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5231116, upper bound: 0.5231142
time: 240.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 548.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 548.60
Output dim: 1, lower bound: -0.5232032, upper bound: 0.5230261
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 548.60
Output dim: 1, lower bound: -0.5231633, upper bound: 0.5230607
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 548.60
Output dim: 1, lower bound: -0.5231633, upper bound: 0.5230237
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 548.60
Output dim: 1, lower bound: -0.5231234, upper bound: 0.5231073
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 548.60
Output dim: 1, lower bound: -0.5231495, upper bound: 0.5230776
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 548.60
Output dim: 1, lower bound: -0.5231119, upper bound: 0.5231160
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 548.60
Output dim: 1, lower bound: -0.5231114, upper bound: 0.5231178
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 548.60
Output dim: 1, lower bound: -0.5230726, upper bound: 0.5231574
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 548.60
Output dim: 1, lower bound: -0.5231520, upper bound: 0.5230729
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 548.60
Output dim: 1, lower bound: -0.5231116, upper bound: 0.5231142
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 548.60
Output dim: 1, lower bound: -0.5231155, upper bound: 0.5231553
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 548.60
Output dim: 1, lower bound: -0.5231040, upper bound: 0.5230662
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 548.60
Output dim: 1, lower bound: -0.5230622, upper bound: 0.5232055
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=4, k_mid=3, eps_mid=0.0117188, abs_max=0.8648518323898315
rel_dist={1: [-0.5234064232429227, 0.5234093100156474]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3115
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3115

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4946725, upper bound: 0.4946400
time: 56.09 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4946356, upper bound: 0.4946707
time: 39.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 95.97 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 95.97
Output dim: 1, lower bound: -0.4946725, upper bound: 0.4946400
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 95.97
Output dim: 1, lower bound: -0.4946356, upper bound: 0.4946707

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122479, 1.4122485
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8500017, 0.8500025
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3458787, 1.3458756
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4644175, 1.4644133
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1700044, 2.1700327
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4195004, 1.4194995
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4046850, 1.4046978
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0929682, 2.0929713
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875769, 0.8875768
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1591035, 1.1591036

Time for backsubstitution: 4.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3113

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4946502, upper bound: 0.4945796
time: 183.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4946128, upper bound: 0.4946173
time: 86.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122486, 1.4122479
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8500024, 0.8500017
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3458754, 1.3458787
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4644132, 1.4644173
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1700325, 2.1700044
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4194994, 1.4195001
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4046977, 1.4046850
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0929711, 2.0929682
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875768, 0.8875768
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1591038, 1.1591034

Time for backsubstitution: 4.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3113

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4946141, upper bound: 0.4946125
time: 15.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945776, upper bound: 0.4946535
time: 120.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 140.33 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 140.33
Output dim: 1, lower bound: -0.4946502, upper bound: 0.4945796
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 140.33
Output dim: 1, lower bound: -0.4946128, upper bound: 0.4946173
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 140.33
Output dim: 1, lower bound: -0.4946141, upper bound: 0.4946125
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 140.33
Output dim: 1, lower bound: -0.4945776, upper bound: 0.4946535

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122684, 1.4122698
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499700, 0.8499709
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3455433, 1.3455415
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4627099, 1.4627032
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1596863, 2.1599059
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4172138, 1.4172194
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4042311, 1.4042456
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0900066, 2.0900288
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875431, 0.8875419
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590956, 1.1590961

Time for backsubstitution: 4.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4946022, upper bound: 0.4945048
time: 21.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945722, upper bound: 0.4945354
time: 35.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122691, 1.4122689
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499700, 0.8499709
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3455447, 1.3455403
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4627073, 1.4627059
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1598771, 2.1597147
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4172200, 1.4172133
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4042330, 1.4042439
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0900257, 2.0900097
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875419, 0.8875430
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590956, 1.1590961

Time for backsubstitution: 4.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945633, upper bound: 0.4945445
time: 35.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945386, upper bound: 0.4945708
time: 56.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122689, 1.4122691
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499709, 0.8499700
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3455402, 1.3455446
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4627059, 1.4627073
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1597149, 2.1598773
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4172133, 1.4172201
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4042439, 1.4042330
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0900099, 2.0900257
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875430, 0.8875419
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590961, 1.1590959

Time for backsubstitution: 4.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945641, upper bound: 0.4945382
time: 109.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945387, upper bound: 0.4945666
time: 67.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122698, 1.4122684
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499707, 0.8499702
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3455414, 1.3455434
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4627032, 1.4627099
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1599057, 2.1596861
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4172195, 1.4172139
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4042456, 1.4042311
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0900285, 2.0900068
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875419, 0.8875431
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590961, 1.1590959

Time for backsubstitution: 4.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945305, upper bound: 0.4945807
time: 44.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945014, upper bound: 0.4945989
time: 20.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 69.60 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 69.60
Output dim: 1, lower bound: -0.4946022, upper bound: 0.4945048
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 69.60
Output dim: 1, lower bound: -0.4945722, upper bound: 0.4945354
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 69.60
Output dim: 1, lower bound: -0.4945633, upper bound: 0.4945445
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 69.60
Output dim: 1, lower bound: -0.4945386, upper bound: 0.4945708
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 69.60
Output dim: 1, lower bound: -0.4945641, upper bound: 0.4945382
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 69.60
Output dim: 1, lower bound: -0.4945387, upper bound: 0.4945666
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 69.60
Output dim: 1, lower bound: -0.4945305, upper bound: 0.4945807
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 69.60
Output dim: 1, lower bound: -0.4945014, upper bound: 0.4945989

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122624, 1.4122641
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499632, 0.8499638
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454825, 1.3454827
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624071, 1.4624119
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1580822, 2.1583946
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168471, 1.4168680
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041197, 1.4041386
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0895782, 2.0896134
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875401, 0.8875382
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590939, 1.1590942

Time for backsubstitution: 4.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4946002, upper bound: 0.4944691
time: 115.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945677, upper bound: 0.4944668
time: 178.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122627, 1.4122639
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499632, 0.8499640
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454846, 1.3454807
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624190, 1.4624001
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1581752, 2.1583023
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168624, 1.4168527
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041240, 1.4041342
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0895915, 2.0896001
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875394, 0.8875388
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590939, 1.1590942

Time for backsubstitution: 4.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945718, upper bound: 0.4944972
time: 18.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945387, upper bound: 0.4945385
time: 14.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122632, 1.4122632
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499632, 0.8499640
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454839, 1.3454815
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624043, 1.4624147
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1582739, 2.1582038
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168533, 1.4168618
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041216, 1.4041368
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0895972, 2.0895944
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875389, 0.8875393
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590939, 1.1590942

Time for backsubstitution: 4.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945631, upper bound: 0.4945050
time: 24.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945297, upper bound: 0.4945406
time: 243.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122634, 1.4122630
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499632, 0.8499640
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454858, 1.3454795
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624162, 1.4624028
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1583660, 2.1581109
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168686, 1.4168465
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041259, 1.4041325
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0896103, 2.0895813
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875382, 0.8875400
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590939, 1.1590942

Time for backsubstitution: 4.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945353, upper bound: 0.4945348
time: 97.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945010, upper bound: 0.4945711
time: 27.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122632, 1.4122634
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499640, 0.8499631
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454796, 1.3454858
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624028, 1.4624159
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1581109, 2.1583660
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168466, 1.4168687
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041326, 1.4041258
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0895813, 2.0896103
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875400, 0.8875382
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590943, 1.1590940

Time for backsubstitution: 4.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945654, upper bound: 0.4945054
time: 18.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945334, upper bound: 0.4945425
time: 41.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122632, 1.4122632
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499639, 0.8499631
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454815, 1.3454838
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624147, 1.4624041
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1582038, 2.1582737
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168619, 1.4168534
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041369, 1.4041215
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0895944, 2.0895972
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875394, 0.8875388
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590943, 1.1590940

Time for backsubstitution: 4.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945377, upper bound: 0.4945363
time: 39.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945026, upper bound: 0.4945687
time: 272.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122639, 1.4122627
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499639, 0.8499631
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454806, 1.3454846
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624000, 1.4624188
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1583025, 2.1581752
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168528, 1.4168625
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041340, 1.4041241
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0896001, 2.0895915
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875388, 0.8875394
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590943, 1.1590940

Time for backsubstitution: 4.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945303, upper bound: 0.4945390
time: 18.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944972, upper bound: 0.4945722
time: 9.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122641, 1.4122624
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499639, 0.8499632
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454827, 1.3454826
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624119, 1.4624069
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1583946, 2.1580825
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168681, 1.4168472
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041383, 1.4041197
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0896134, 2.0895782
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875382, 0.8875400
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590943, 1.1590940

Time for backsubstitution: 4.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3397
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3397

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945016, upper bound: 0.4945697
time: 18.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944666, upper bound: 0.4946023
time: 18.03 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 41.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4946002, upper bound: 0.4944691
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945677, upper bound: 0.4944668
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945718, upper bound: 0.4944972
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945387, upper bound: 0.4945385
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945631, upper bound: 0.4945050
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945297, upper bound: 0.4945406
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945353, upper bound: 0.4945348
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945010, upper bound: 0.4945711
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945654, upper bound: 0.4945054
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945334, upper bound: 0.4945425
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945377, upper bound: 0.4945363
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945026, upper bound: 0.4945687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945303, upper bound: 0.4945390
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4944972, upper bound: 0.4945722
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4945016, upper bound: 0.4945697
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.03
Output dim: 1, lower bound: -0.4944666, upper bound: 0.4946023

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122624, 1.4122641
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499593, 0.8499599
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454832, 1.3454833
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624120, 1.4624171
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1580858, 2.1583977
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168534, 1.4168742
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041073, 1.4041264
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0895796, 2.0896144
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875401, 0.8875382
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590980, 1.1590985

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945748, upper bound: 0.4944322
time: 55.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945478, upper bound: 0.4944423
time: 305.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122624, 1.4122641
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499593, 0.8499599
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454832, 1.3454833
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624120, 1.4624171
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1580858, 2.1583977
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168534, 1.4168742
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041076, 1.4041262
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0895796, 2.0896144
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875401, 0.8875382
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590983, 1.1590983

Time for backsubstitution: 4.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945443, upper bound: 0.4944667
time: 18.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945144, upper bound: 0.4944746
time: 58.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122629, 1.4122639
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499593, 0.8499599
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454853, 1.3454813
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624239, 1.4624052
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1581783, 2.1583056
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168687, 1.4168589
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041116, 1.4041221
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0895925, 2.0896013
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875394, 0.8875388
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590980, 1.1590985

Time for backsubstitution: 4.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945430, upper bound: 0.4944444
time: 456.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945341, upper bound: 0.4944732
time: 14.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122629, 1.4122639
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499591, 0.8499599
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454853, 1.3454813
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624239, 1.4624053
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1581783, 2.1583056
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168687, 1.4168589
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041121, 1.4041219
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0895925, 2.0896013
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875394, 0.8875388
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590983, 1.1590983

Time for backsubstitution: 4.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945072, upper bound: 0.4944790
time: 19.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945000, upper bound: 0.4945042
time: 192.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122634, 1.4122634
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499591, 0.8499599
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454846, 1.3454821
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624094, 1.4624199
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1582770, 2.1582069
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168596, 1.4168680
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041090, 1.4041247
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0895982, 2.0895956
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875389, 0.8875393
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590980, 1.1590985

Time for backsubstitution: 4.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945393, upper bound: 0.4944733
time: 39.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945146, upper bound: 0.4944794
time: 340.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.9586557, 0.7209685, -0.9586557, 0.7209685, -1.4122634, 1.4122634
1: -1.0997157, 0.6295103, -1.0997157, 0.6295103, -0.8499591, 0.8499599
2: -3.6659455, -1.6741122, -3.6659455, -1.6741122, -1.3454846, 1.3454821
3: -4.1973753, -0.7891464, -4.1973753, -0.7891464, -1.4624091, 1.4624200
4: -5.0275388, -1.6263011, -5.0275388, -1.6263011, -2.1582770, 2.1582069
5: -4.3328991, -1.0012889, -4.3328991, -1.0012889, -1.4168596, 1.4168680
6: -8.4144001, -4.5927444, -8.4144001, -4.5927444, -1.4041095, 1.4041244
7: -4.6260777, -1.2471528, -4.6260777, -1.2471528, -2.0895982, 2.0895956
8: -0.1772420, 0.7755450, -0.1772420, 0.7755450, -0.8875389, 0.8875393
9: -1.5128123, 0.1989471, -1.5128123, 0.1989471, -1.1590983, 1.1590983

Time for backsubstitution: 4.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 3116
type: RSZ, layer: 1, pos: 3398
type: RSZ, layer: 1, pos: 2453
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3413
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2425
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 3399
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 3099
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2200
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 474
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 473
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3440
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 489
type: RSZ, layer: 1, pos: 3101
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 505
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2482
type: RSZ, layer: 1, pos: 3201
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3184
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2931
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 3226
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 475
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 2486
type: RSZ, layer: 1, pos: 2948
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3347
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2481
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 501
type: RSZ, layer: 1, pos: 427
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3348
type: RSZ, layer: 1, pos: 3345
type: RSZ, layer: 1, pos: 2606
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3565
type: RSZ, layer: 1, pos: 3174
type: RSZ, layer: 1, pos: 3450
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 2621
type: RSZ, layer: 1, pos: 3346
type: RSZ, layer: 1, pos: 2289
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2577
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3187
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 3458
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2674
type: RSZ, layer: 1, pos: 3091
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2483
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3175
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3341
type: RSZ, layer: 1, pos: 2379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3121
type: RSZ, layer: 1, pos: 2427
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2300
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 3379
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2989
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 297
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3327
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2368
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2533
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 464
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 2054
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2149
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2161
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 2267
type: RSZ, layer: 1, pos: 2279
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2471
type: RSZ, layer: 1, pos: 2472
type: RSZ, layer: 1, pos: 2473
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2493
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2598
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2688
type: RSZ, layer: 1, pos: 2689
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3029
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3065
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3137
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3368
type: RSZ, layer: 1, pos: 3494
type: RSZ, layer: 1, pos: 3589

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2440

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4945083, upper bound: 0.4945047
time: 238.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4944766, upper bound: 0.4945090
time: 126.46 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 369.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4945748, upper bound: 0.4944322
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4945478, upper bound: 0.4944423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4945443, upper bound: 0.4944667
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4945144, upper bound: 0.4944746
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4945430, upper bound: 0.4944444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4945341, upper bound: 0.4944732
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4945072, upper bound: 0.4944790
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4945000, upper bound: 0.4945042
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4945393, upper bound: 0.4944733
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4945146, upper bound: 0.4944794
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4945083, upper bound: 0.4945047
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 369.55
Output dim: 1, lower bound: -0.4944766, upper bound: 0.4945090
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 369.55
Output dim: 1, lower bound: -0.4945353, upper bound: 0.4945348
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 369.55
Output dim: 1, lower bound: -0.4945010, upper bound: 0.4945711
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 369.55
Output dim: 1, lower bound: -0.4945654, upper bound: 0.4945054
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 369.55
Output dim: 1, lower bound: -0.4945334, upper bound: 0.4945425
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 369.55
Output dim: 1, lower bound: -0.4945377, upper bound: 0.4945363
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 369.55
Output dim: 1, lower bound: -0.4945026, upper bound: 0.4945687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 369.55
Output dim: 1, lower bound: -0.4945303, upper bound: 0.4945390
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 369.55
Output dim: 1, lower bound: -0.4944972, upper bound: 0.4945722
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 369.55
Output dim: 1, lower bound: -0.4945016, upper bound: 0.4945697
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 369.55
Output dim: 1, lower bound: -0.4944666, upper bound: 0.4946023
Binary search (step 2): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.8498140573501587
rel_dist={1: [-0.49469755591069386, 0.49469970537319075]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 12391.83 seconds
