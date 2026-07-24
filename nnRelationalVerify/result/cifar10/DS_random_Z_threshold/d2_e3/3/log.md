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
execution time: IAR + RelationalAnalysis = 7.26 + 169.76 = 177.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1700003, upper bound: 0.1700039

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 525

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2449

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699093, upper bound: 0.1699141
time: 195.05 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699109, upper bound: 0.1699136
time: 9.46 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 204.53 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 204.53
Output dim: 1, lower bound: -0.1699093, upper bound: 0.1699141
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 204.53
Output dim: 1, lower bound: -0.1699109, upper bound: 0.1699136

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9008631, 0.9008633
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545618, 0.3545685
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9468712, 0.9468759
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8933685, 0.8933886
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4328589, 1.4328842
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.8009549, 0.8009771
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9308280, 0.9308393
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3482468, 1.3483220
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214688, 0.5214684
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594674, 0.5594676

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2381

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2373

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1695545, upper bound: 0.1695563
time: 303.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1695546, upper bound: 0.1695566
time: 124.93 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9008633, 0.9008630
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545684, 0.3545618
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9468759, 0.9468713
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8933885, 0.8933684
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4328842, 1.4328588
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.8009771, 0.8009549
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9308394, 0.9308280
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3483222, 1.3482468
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214684, 0.5214689
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594676, 0.5594674

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2373

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 525

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699091, upper bound: 0.1699048
time: 51.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699082, upper bound: 0.1699100
time: 33.53 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 90.82 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 90.82
Output dim: 1, lower bound: -0.1695545, upper bound: 0.1695563
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 90.82
Output dim: 1, lower bound: -0.1695546, upper bound: 0.1695566
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 90.82
Output dim: 1, lower bound: -0.1699091, upper bound: 0.1699048
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 90.82
Output dim: 1, lower bound: -0.1699082, upper bound: 0.1699100

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9008632, 0.9008628
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545687, 0.3545620
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9468782, 0.9468736
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8933573, 0.8933380
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4328780, 1.4328526
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.8009769, 0.8009546
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9308100, 0.9308003
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3482559, 1.3481784
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214697, 0.5214701
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594667, 0.5594665

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 3519

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2640

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1697912, upper bound: 0.1697881
time: 83.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1697897, upper bound: 0.1697889
time: 14.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9008632, 0.9008628
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545686, 0.3545620
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9468782, 0.9468735
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8933582, 0.8933370
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4328780, 1.4328524
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.8009769, 0.8009546
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9308116, 0.9307986
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3482535, 1.3481807
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214696, 0.5214701
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594667, 0.5594666

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2615

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699023, upper bound: 0.1699045
time: 16.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699023, upper bound: 0.1699045
time: 16.90 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 39.61 seconds
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 39.61
Output dim: 1, lower bound: -0.1697912, upper bound: 0.1697881
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 39.61
Output dim: 1, lower bound: -0.1697897, upper bound: 0.1697889
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.61
Output dim: 1, lower bound: -0.1699023, upper bound: 0.1699045
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.61
Output dim: 1, lower bound: -0.1699023, upper bound: 0.1699045

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9008486, 0.9008470
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545608, 0.3545542
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9468412, 0.9468408
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8918524, 0.8920150
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4324772, 1.4325173
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7995597, 0.7997041
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9300656, 0.9301149
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3473754, 1.3473936
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214660, 0.5214665
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594650, 0.5594649

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3348

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2075

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1697760, upper bound: 0.1697775
time: 221.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698964, upper bound: 0.1697777
time: 162.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9008472, 0.9008484
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545608, 0.3545541
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9468455, 0.9468366
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8920362, 0.8918310
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4325428, 1.4324516
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7997264, 0.7995374
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9301281, 0.9300526
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3474665, 1.3473026
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214660, 0.5214665
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594651, 0.5594649

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2969

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2186

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698987, upper bound: 0.1699004
time: 490.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698990, upper bound: 0.1699015
time: 12.09 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 508.29 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 508.29
Output dim: 1, lower bound: -0.1697760, upper bound: 0.1697775
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 508.29
Output dim: 1, lower bound: -0.1698964, upper bound: 0.1697777
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 508.29
Output dim: 1, lower bound: -0.1698987, upper bound: 0.1699004
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 508.29
Output dim: 1, lower bound: -0.1698990, upper bound: 0.1699015

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007818, 0.9007877
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3532542, 0.3532071
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9468415, 0.9468409
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8900537, 0.8902194
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4323297, 1.4323798
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7981381, 0.7982858
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9282377, 0.9282429
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3471308, 1.3471485
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214585, 0.5214585
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5587372, 0.5587375

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3438

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2382

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1695914, upper bound: 0.1694753
time: 8.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1695914, upper bound: 0.1694753
time: 7.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007832, 0.9007822
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545459, 0.3545393
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9466990, 0.9466884
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8866820, 0.8866785
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4304442, 1.4304365
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7943684, 0.7943803
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9277632, 0.9277750
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3427546, 1.3427041
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214875, 0.5214880
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594642, 0.5594642

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3087

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2674

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1698095, upper bound: 0.1698132
time: 12.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1698100, upper bound: 0.1698136
time: 9.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007810, 0.9007843
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545459, 0.3545392
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9466974, 0.9466900
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8868837, 0.8864768
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4305277, 1.4303533
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7945713, 0.7941794
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9278510, 0.9276881
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3428690, 1.3425906
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214875, 0.5214880
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594643, 0.5594641

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2238

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3399

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698955, upper bound: 0.1698986
time: 8.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698971, upper bound: 0.1698963
time: 45.65 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 60.37 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 60.37
Output dim: 1, lower bound: -0.1695914, upper bound: 0.1694753
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 60.37
Output dim: 1, lower bound: -0.1695914, upper bound: 0.1694753
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 60.37
Output dim: 1, lower bound: -0.1698095, upper bound: 0.1698132
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 60.37
Output dim: 1, lower bound: -0.1698100, upper bound: 0.1698136
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 60.37
Output dim: 1, lower bound: -0.1698955, upper bound: 0.1698986
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 60.37
Output dim: 1, lower bound: -0.1698971, upper bound: 0.1698963

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007787, 0.9007819
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545433, 0.3545365
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9466990, 0.9466916
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8868852, 0.8864782
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4305315, 1.4303572
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7945725, 0.7941808
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9278512, 0.9276881
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3428692, 1.3425905
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214874, 0.5214878
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594699, 0.5594696

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 91

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3137

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698952, upper bound: 0.1698987
time: 138.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698951, upper bound: 0.1698973
time: 12.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007787, 0.9007819
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545433, 0.3545366
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9466990, 0.9466916
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8868852, 0.8864782
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4305317, 1.4303570
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7945725, 0.7941808
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9278512, 0.9276881
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3428692, 1.3425905
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214874, 0.5214878
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594699, 0.5594696

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2075

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2603

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1696233, upper bound: 0.1696225
time: 13.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1696233, upper bound: 0.1696219
time: 15.96 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 35.22 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 35.22
Output dim: 1, lower bound: -0.1698952, upper bound: 0.1698987
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 35.22
Output dim: 1, lower bound: -0.1698951, upper bound: 0.1698973
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 35.22
Output dim: 1, lower bound: -0.1696233, upper bound: 0.1696225
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 35.22
Output dim: 1, lower bound: -0.1696233, upper bound: 0.1696219

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007787, 0.9007819
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545433, 0.3545365
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9466990, 0.9466916
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8868852, 0.8864782
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4305315, 1.4303572
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7945725, 0.7941808
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9278512, 0.9276881
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3428692, 1.3425905
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214874, 0.5214878
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594699, 0.5594696

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 464

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698954, upper bound: 0.1699009
time: 7.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698954, upper bound: 0.1698990
time: 214.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007787, 0.9007819
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545433, 0.3545365
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9466990, 0.9466916
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8868852, 0.8864782
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4305315, 1.4303572
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7945725, 0.7941808
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9278512, 0.9276881
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3428692, 1.3425905
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214874, 0.5214878
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594699, 0.5594696

Time for backsubstitution: 6.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3367

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698953, upper bound: 0.1698991
time: 146.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698953, upper bound: 0.1698948
time: 82.52 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 235.31 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 235.31
Output dim: 1, lower bound: -0.1698954, upper bound: 0.1699009
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 235.31
Output dim: 1, lower bound: -0.1698954, upper bound: 0.1698990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 235.31
Output dim: 1, lower bound: -0.1698953, upper bound: 0.1698991
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 235.31
Output dim: 1, lower bound: -0.1698953, upper bound: 0.1698948

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007787, 0.9007819
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545433, 0.3545365
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9466990, 0.9466916
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8868852, 0.8864782
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4305315, 1.4303572
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7945725, 0.7941808
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9278512, 0.9276881
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3428692, 1.3425905
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214874, 0.5214878
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594699, 0.5594696

Time for backsubstitution: 6.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 445

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2238

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1697852, upper bound: 0.1697879
time: 187.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1697852, upper bound: 0.1697858
time: 164.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007787, 0.9007819
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545433, 0.3545365
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9466990, 0.9466916
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8868852, 0.8864782
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4305315, 1.4303572
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7945725, 0.7941808
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9278512, 0.9276881
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3428692, 1.3425905
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214874, 0.5214878
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594699, 0.5594696

Time for backsubstitution: 6.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2689

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698760, upper bound: 0.1698807
time: 12.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698760, upper bound: 0.1698798
time: 156.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007787, 0.9007819
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545433, 0.3545365
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9466990, 0.9466916
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8868852, 0.8864782
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4305315, 1.4303572
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7945725, 0.7941808
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9278512, 0.9276881
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3428692, 1.3425905
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214874, 0.5214878
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594699, 0.5594696

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2618

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2440

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698830, upper bound: 0.1698859
time: 5.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698801, upper bound: 0.1698865
time: 146.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007787, 0.9007819
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545433, 0.3545365
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9466990, 0.9466916
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8868852, 0.8864782
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4305315, 1.4303572
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7945725, 0.7941808
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9278512, 0.9276881
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3428692, 1.3425905
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214874, 0.5214878
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594699, 0.5594696

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 464
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2969

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698903, upper bound: 0.1698995
time: 15.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698903, upper bound: 0.1698912
time: 162.30 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 184.29 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 184.29
Output dim: 1, lower bound: -0.1697852, upper bound: 0.1697879
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 184.29
Output dim: 1, lower bound: -0.1697852, upper bound: 0.1697858
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 184.29
Output dim: 1, lower bound: -0.1698760, upper bound: 0.1698807
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 184.29
Output dim: 1, lower bound: -0.1698760, upper bound: 0.1698798
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 184.29
Output dim: 1, lower bound: -0.1698830, upper bound: 0.1698859
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 184.29
Output dim: 1, lower bound: -0.1698801, upper bound: 0.1698865
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 184.29
Output dim: 1, lower bound: -0.1698903, upper bound: 0.1698995
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 184.29
Output dim: 1, lower bound: -0.1698903, upper bound: 0.1698912

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6456727, 0.5496855, -0.6456727, 0.5496855, -0.9007787, 0.9007819
1: -0.6780654, 0.4143413, -0.6780654, 0.4143413, -0.3545433, 0.3545365
2: -3.3954582, -1.9495008, -3.3954582, -1.9495008, -0.9466990, 0.9466916
3: -3.7291336, -1.2488751, -3.7291336, -1.2488751, -0.8868852, 0.8864782
4: -4.5492096, -2.0701752, -4.5492096, -2.0701752, -1.4305315, 1.4303572
5: -3.8423536, -1.4658918, -3.8423536, -1.4658918, -0.7945725, 0.7941808
6: -7.9546523, -5.1249242, -7.9546523, -5.1249242, -0.9278512, 0.9276881
7: -3.9798756, -1.6032474, -3.9798756, -1.6032474, -1.3428692, 1.3425905
8: 0.0570735, 0.6355705, 0.0570735, 0.6355705, -0.5214874, 0.5214878
9: -1.0491056, 0.0330073, -1.0491056, 0.0330073, -0.5594699, 0.5594696

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 646
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 475
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3397
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 3090

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 761

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698637, upper bound: 0.1698732
time: 187.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698699, upper bound: 0.1698676
time: 12.31 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 205.96 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 205.96
Output dim: 1, lower bound: -0.1698637, upper bound: 0.1698732
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 205.96
Output dim: 1, lower bound: -0.1698699, upper bound: 0.1698676
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 205.96
Output dim: 1, lower bound: -0.1698760, upper bound: 0.1698798
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 205.96
Output dim: 1, lower bound: -0.1698830, upper bound: 0.1698859
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 205.96
Output dim: 1, lower bound: -0.1698801, upper bound: 0.1698865
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 205.96
Output dim: 1, lower bound: -0.1698903, upper bound: 0.1698995
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 205.96
Output dim: 1, lower bound: -0.1698903, upper bound: 0.1698912

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 177.01 + 3619.97 = 3796.98 seconds
