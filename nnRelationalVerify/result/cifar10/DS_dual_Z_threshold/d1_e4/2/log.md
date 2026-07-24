## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0497259243


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3612083, 0.3612083)
1: (-0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632463, 0.3632463)
2: (-1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183412, 0.3183412)
3: (-4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419307, 0.4419307)
4: (-1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223641, 0.5223641)
5: (-4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168744, 0.4168744)
6: (-5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121849, 0.3121849)
7: (-0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559079, 0.9559078)
8: (-2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1992245, 1.1992247)
9: (-0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419469, 0.6419468)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.80 + 64.64 = 72.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0497747, upper bound: 0.0497754

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 302

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497747, upper bound: 0.0497563
time: 5.41 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497562, upper bound: 0.0497764
time: 62.70 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 68.18 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 68.18
Output dim: 1, lower bound: -0.0497747, upper bound: 0.0497563
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 68.18
Output dim: 1, lower bound: -0.0497562, upper bound: 0.0497764

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3607655, 0.3607291
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3630958, 0.3630863
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173290, 0.3173952
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4415627, 0.4415857
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223516, 0.5223539
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4164315, 0.4164648
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3107170, 0.3108102
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9558395, 0.9558456
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1989436, 1.1989276
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6416702, 0.6416733

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2510

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497754, upper bound: 0.0497472
time: 152.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497669, upper bound: 0.0497564
time: 5.77 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3607290, 0.3607655
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3630863, 0.3630958
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173952, 0.3173290
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4415857, 0.4415627
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223539, 0.5223516
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4164648, 0.4164315
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3108102, 0.3107170
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9558456, 0.9558394
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1989274, 1.1989439
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6416733, 0.6416702

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2510

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497525, upper bound: 0.0497669
time: 16.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497453, upper bound: 0.0497755
time: 52.47 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 74.94 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 74.94
Output dim: 1, lower bound: -0.0497754, upper bound: 0.0497472
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 74.94
Output dim: 1, lower bound: -0.0497669, upper bound: 0.0497564
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 74.94
Output dim: 1, lower bound: -0.0497525, upper bound: 0.0497669
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 74.94
Output dim: 1, lower bound: -0.0497453, upper bound: 0.0497755

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3604889, 0.3604541
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3630893, 0.3630796
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3172818, 0.3173442
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4403961, 0.4404102
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223490, 0.5223514
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142511, 0.4142781
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3100905, 0.3101803
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9555726, 0.9555774
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986575, 1.1986430
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6416783, 0.6416814

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 312

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497751, upper bound: 0.0497096
time: 212.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497377, upper bound: 0.0497467
time: 5.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3604904, 0.3604525
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3630890, 0.3630798
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3172781, 0.3173479
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4403872, 0.4404191
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223490, 0.5223514
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142447, 0.4142845
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3100871, 0.3101837
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9555713, 0.9555786
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986589, 1.1986415
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6416783, 0.6416814

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 312

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497672, upper bound: 0.0497171
time: 182.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497305, upper bound: 0.0497561
time: 75.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3604525, 0.3604904
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3630799, 0.3630889
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173479, 0.3172781
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4404191, 0.4403872
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223514, 0.5223490
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142845, 0.4142447
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3101837, 0.3100871
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9555787, 0.9555714
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986413, 1.1986592
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6416817, 0.6416781

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 312

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497566, upper bound: 0.0497097
time: 127.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497182, upper bound: 0.0497676
time: 141.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3604541, 0.3604890
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3630797, 0.3630894
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173442, 0.3172818
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4404102, 0.4403962
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223514, 0.5223490
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142780, 0.4142512
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3101803, 0.3100905
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9555774, 0.9555724
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986427, 1.1986578
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6416817, 0.6416781

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 312

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497469, upper bound: 0.0497389
time: 69.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497098, upper bound: 0.0497347
time: 74.28 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 149.37 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 149.37
Output dim: 1, lower bound: -0.0497751, upper bound: 0.0497096
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 149.37
Output dim: 1, lower bound: -0.0497377, upper bound: 0.0497467
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 149.37
Output dim: 1, lower bound: -0.0497672, upper bound: 0.0497171
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 149.37
Output dim: 1, lower bound: -0.0497305, upper bound: 0.0497561
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 149.37
Output dim: 1, lower bound: -0.0497566, upper bound: 0.0497097
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 149.37
Output dim: 1, lower bound: -0.0497182, upper bound: 0.0497676
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 149.37
Output dim: 1, lower bound: -0.0497469, upper bound: 0.0497389
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 149.37
Output dim: 1, lower bound: -0.0497098, upper bound: 0.0497347

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3603054, 0.3602640
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3631420, 0.3631321
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173046, 0.3173684
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4403566, 0.4403723
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5224454, 0.5224395
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142241, 0.4142528
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3099114, 0.3100072
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9556420, 0.9556514
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986265, 1.1986108
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6418426, 0.6418520

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 295

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497752, upper bound: 0.0496617
time: 82.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497267, upper bound: 0.0497091
time: 7.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3602989, 0.3602706
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3631417, 0.3631323
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173059, 0.3173671
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4403582, 0.4403706
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5224371, 0.5224478
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142260, 0.4142510
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3099173, 0.3100012
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9556465, 0.9556469
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986256, 1.1986113
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6418486, 0.6418458

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 295

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497381, upper bound: 0.0496987
time: 7.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496891, upper bound: 0.0497097
time: 48.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3603070, 0.3602626
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3631417, 0.3631323
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173009, 0.3173721
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4403477, 0.4403812
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5224454, 0.5224395
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142177, 0.4142593
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3099080, 0.3100106
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9556407, 0.9556526
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986277, 1.1986094
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6418426, 0.6418520

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 295

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497664, upper bound: 0.0496699
time: 12.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0497183, upper bound: 0.0497184
time: 5.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3603004, 0.3602690
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3631414, 0.3631325
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173022, 0.3173707
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4403493, 0.4403795
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5224371, 0.5224478
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142196, 0.4142573
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3099140, 0.3100046
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9556453, 0.9556481
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986270, 1.1986098
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6418486, 0.6418458

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 295

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497284, upper bound: 0.0497076
time: 17.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496798, upper bound: 0.0497558
time: 6.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3602691, 0.3603005
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3631325, 0.3631414
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173708, 0.3173022
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4403795, 0.4403493
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5224478, 0.5224371
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142574, 0.4142195
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3100046, 0.3099140
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9556481, 0.9556452
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986101, 1.1986270
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6418457, 0.6418487

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 295

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497549, upper bound: 0.0496816
time: 6.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497053, upper bound: 0.0497297
time: 57.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3602625, 0.3603069
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3631322, 0.3631416
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173721, 0.3173009
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4403812, 0.4403477
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5224394, 0.5224454
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142593, 0.4142177
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3100106, 0.3099080
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9556527, 0.9556407
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986091, 1.1986279
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6418519, 0.6418425

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 295

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0497157, upper bound: 0.0497188
time: 6.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496690, upper bound: 0.0497185
time: 106.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3602706, 0.3602989
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3631323, 0.3631416
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173671, 0.3173059
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4403706, 0.4403583
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5224478, 0.5224371
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142510, 0.4142259
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3100012, 0.3099173
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9556468, 0.9556464
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986113, 1.1986256
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6418457, 0.6418487

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 295

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497456, upper bound: 0.0496615
time: 58.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496978, upper bound: 0.0497378
time: 17.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3602639, 0.3603055
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3631320, 0.3631418
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3173684, 0.3173046
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4403722, 0.4403566
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5224394, 0.5224454
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4142529, 0.4142240
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3100072, 0.3099114
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9556514, 0.9556421
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986105, 1.1986265
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6418519, 0.6418425

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 295

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497080, upper bound: 0.0497272
time: 33.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496588, upper bound: 0.0497754
time: 5.20 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 44.94 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0497752, upper bound: 0.0496617
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0497267, upper bound: 0.0497091
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0497381, upper bound: 0.0496987
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0496891, upper bound: 0.0497097
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0497664, upper bound: 0.0496699
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0497183, upper bound: 0.0497184
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0497284, upper bound: 0.0497076
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0496798, upper bound: 0.0497558
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0497549, upper bound: 0.0496816
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0497053, upper bound: 0.0497297
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0497157, upper bound: 0.0497188
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0496690, upper bound: 0.0497185
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0497456, upper bound: 0.0496615
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0496978, upper bound: 0.0497378
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0497080, upper bound: 0.0497272
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.94
Output dim: 1, lower bound: -0.0496588, upper bound: 0.0497754

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 72.43 + 1747.62 = 1820.06 seconds
