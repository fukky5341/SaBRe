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
execution time: IAR + RelationalAnalysis = 7.78 + 167.90 = 175.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1700003, upper bound: 0.1700039

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3589

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3113

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699934, upper bound: 0.1699883
time: 226.69 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699863, upper bound: 0.1699976
time: 176.52 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 403.29 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 403.29
Output dim: 1, lower bound: -0.1699934, upper bound: 0.1699883
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 403.29
Output dim: 1, lower bound: -0.1699863, upper bound: 0.1699976

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9008934, 0.9008934
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3546113, 0.3546113
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9467547, 0.9467558
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8926656, 0.8926602
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4256799, 1.4258653
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7999712, 0.7999783
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9308084, 0.9308106
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3465872, 1.3466065
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214605, 0.5214593
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594594, 0.5594594

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3589

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2439

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1696654, upper bound: 0.1696551
time: 12.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1696590, upper bound: 0.1696606
time: 200.63 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9008934, 0.9008933
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3546113, 0.3546113
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9467556, 0.9467546
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8926601, 0.8926656
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4258652, 1.4256799
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7999783, 0.7999713
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9308103, 0.9308084
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3466065, 1.3465873
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214593, 0.5214605
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594594, 0.5594594

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3589

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2439

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1696583, upper bound: 0.1696626
time: 12.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1696519, upper bound: 0.1696682
time: 18.49 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 37.41 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 37.41
Output dim: 1, lower bound: -0.1696654, upper bound: 0.1696551
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 37.41
Output dim: 1, lower bound: -0.1696590, upper bound: 0.1696606
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 37.41
Output dim: 1, lower bound: -0.1696583, upper bound: 0.1696626
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 37.41
Output dim: 1, lower bound: -0.1696519, upper bound: 0.1696682

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 175.68 + 659.48 = 835.16 seconds
