## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.1698326973


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9008808, 0.9008807)
1: (-0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3546365, 0.3546365)
2: (-3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9470021, 0.9470022)
3: (-3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8938813, 0.8938815)
4: (-4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4336028, 1.4336029)
5: (-3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.8016400, 0.8016401)
6: (-7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9311599, 0.9311598)
7: (-3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3487809, 1.3487806)
8: (0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214881, 0.5214880)
9: (-1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594689, 0.5594690)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.42 + 170.56 = 178.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1700003, upper bound: 0.1700039

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2654
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2080

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1697284, upper bound: 0.1699662
time: 11.98 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699623, upper bound: 0.1699630
time: 585.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 597.83 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 597.83
Output dim: 1, lower bound: -0.1697284, upper bound: 0.1699662
NS_A2, status: Status.UNKNOWN, split count: 1, time: 597.83
Output dim: 1, lower bound: -0.1699623, upper bound: 0.1699630

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.6454960, 0.5494189, -0.6456549, 0.5494534, -0.9004969, 0.9006001
1: -0.6759505, 0.4127024, -0.6762899, 0.4143370, -0.3527918, 0.3518375
2: -3.3954456, -1.9495127, -3.3954518, -1.9495080, -0.9469783, 0.9469579
3: -3.7288952, -1.2489989, -3.7289274, -1.2488774, -0.8936625, 0.8937120
4: -4.5489068, -2.0707805, -4.5491900, -2.0707073, -1.4328766, 1.4330508
5: -3.8421521, -1.4659425, -3.8422089, -1.4658957, -0.8013873, 0.8013312
6: -7.9537811, -5.1256123, -7.9538889, -5.1249285, -0.9306152, 0.9306632
7: -3.9794998, -1.6033273, -3.9798551, -1.6033146, -1.3483615, 1.3486894
8: 0.0571087, 0.6355516, 0.0570992, 0.6355650, -0.5214249, 0.5214234
9: -1.0479357, 0.0320330, -1.0481509, 0.0330055, -0.5583860, 0.5579996

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2654
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

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2522

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1694104, upper bound: 0.1696902
time: 10.77 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1696070, upper bound: 0.1698411
time: 411.27 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.6456599, 0.5495544, -0.6456620, 0.5495707, -0.9007621, 0.9005705
1: -0.6777411, 0.4143382, -0.6777818, 0.4143387, -0.3518094, 0.3541857
2: -3.3954554, -1.9495051, -3.3954558, -1.9495044, -0.9469805, 0.9469980
3: -3.7290912, -1.2488770, -3.7290964, -1.2488768, -0.8937056, 0.8938017
4: -4.5491962, -2.0704513, -4.5491972, -2.0704207, -1.4333234, 1.4328151
5: -3.8422570, -1.4658942, -3.8422685, -1.4658945, -0.8013247, 0.8015540
6: -7.9545078, -5.1249242, -7.9545212, -5.1249247, -0.9306619, 0.9307302
7: -3.9798667, -1.6032963, -3.9798672, -1.6032904, -1.3487275, 1.3486795
8: 0.0570897, 0.6355662, 0.0570876, 0.6355668, -0.5214525, 0.5214493
9: -1.0489856, 0.0330058, -1.0489999, 0.0330059, -0.5579510, 0.5592406

Time for backsubstitution: 6.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2533
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3397
type: B, layer: 1, pos: 3347
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 475
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 464
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2654
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

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2522

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1696445, upper bound: 0.1696880
time: 91.18 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698412, upper bound: 0.1698415
time: 211.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 309.51 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 309.51
Output dim: 1, lower bound: -0.1694104, upper bound: 0.1696902
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 309.51
Output dim: 1, lower bound: -0.1696070, upper bound: 0.1698411
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 309.51
Output dim: 1, lower bound: -0.1696445, upper bound: 0.1696880
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 309.51
Output dim: 1, lower bound: -0.1698412, upper bound: 0.1698415

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.6454192, 0.5494186, -0.6455559, 0.5494531, -0.9003979, 0.9003106
1: -0.6759204, 0.4122159, -0.6762511, 0.4137069, -0.3469873, 0.3517143
2: -3.3954117, -1.9495153, -3.3954074, -1.9495099, -0.9469330, 0.9469139
3: -3.7288928, -1.2490789, -3.7289238, -1.2489822, -0.8932344, 0.8936570
4: -4.5487208, -2.0707829, -4.5489502, -2.0707116, -1.4327037, 1.4322634
5: -3.8421402, -1.4660122, -3.8421931, -1.4659817, -0.8006819, 0.8012674
6: -7.9537802, -5.1257677, -7.9538875, -5.1251235, -0.9297014, 0.9306020
7: -3.9794128, -1.6033270, -3.9797440, -1.6033156, -1.3482897, 1.3485949
8: 0.0571378, 0.6355458, 0.0571367, 0.6355575, -0.5214011, 0.5213712
9: -1.0479064, 0.0318490, -1.0481136, 0.0327681, -0.5554634, 0.5579296

Time for backsubstitution: 6.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2654
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2493

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1694540, upper bound: 0.1696887
time: 15.82 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1694540, upper bound: 0.1696887
time: 78.53 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.6455839, 0.5495540, -0.6455628, 0.5495704, -0.9006634, 0.9002813
1: -0.6777106, 0.4138516, -0.6777435, 0.4137077, -0.3460051, 0.3540625
2: -3.3954213, -1.9495065, -3.3954115, -1.9495063, -0.9469348, 0.9469544
3: -3.7290897, -1.2489572, -3.7290936, -1.2489810, -0.8932776, 0.8937467
4: -4.5490108, -2.0704546, -4.5489578, -2.0704248, -1.4331503, 1.4320273
5: -3.8422449, -1.4659632, -3.8422518, -1.4659808, -0.8006196, 0.8014902
6: -7.9545074, -5.1250782, -7.9545197, -5.1251202, -0.9297481, 0.9306687
7: -3.9797807, -1.6032975, -3.9797561, -1.6032913, -1.3486564, 1.3485847
8: 0.0571186, 0.6355604, 0.0571253, 0.6355593, -0.5214286, 0.5213969
9: -1.0489562, 0.0328219, -1.0489627, 0.0327686, -0.5550284, 0.5591710

Time for backsubstitution: 6.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2533
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3397
type: A, layer: 1, pos: 3347
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 475
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 464
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2654
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2493

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1696882, upper bound: 0.1696892
time: 7.83 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1696883, upper bound: 0.1696908
time: 178.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 193.14 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 193.14
Output dim: 1, lower bound: -0.1694540, upper bound: 0.1696887
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 193.14
Output dim: 1, lower bound: -0.1694540, upper bound: 0.1696887
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 193.14
Output dim: 1, lower bound: -0.1696882, upper bound: 0.1696892
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 193.14
Output dim: 1, lower bound: -0.1696883, upper bound: 0.1696908

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 178.99 + 1629.72 = 1808.71 seconds
