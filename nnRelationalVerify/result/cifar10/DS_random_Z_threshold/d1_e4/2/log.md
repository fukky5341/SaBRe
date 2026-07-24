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
execution time: IAR + RelationalAnalysis = 8.30 + 67.29 = 75.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0497747, upper bound: 0.0497754

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2622

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2253

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497732, upper bound: 0.0497659
time: 5.43 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497638, upper bound: 0.0497755
time: 16.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 22.03 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 22.03
Output dim: 1, lower bound: -0.0497732, upper bound: 0.0497659
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 22.03
Output dim: 1, lower bound: -0.0497638, upper bound: 0.0497755

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3611990, 0.3612005
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632458, 0.3632456
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183390, 0.3183388
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419267, 0.4419261
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223621, 0.5223625
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168687, 0.4168679
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121765, 0.3121831
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559077, 0.9559073
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1992178, 1.1992185
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419466, 0.6419463

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1015

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497719, upper bound: 0.0497539
time: 59.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497638, upper bound: 0.0497657
time: 63.14 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3612004, 0.3611990
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632455, 0.3632457
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183388, 0.3183390
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419261, 0.4419267
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223625, 0.5223620
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168679, 0.4168687
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121830, 0.3121765
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559075, 0.9559078
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1992188, 1.1992176
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419463, 0.6419466

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2048

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2658

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497331, upper bound: 0.0497513
time: 37.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497412, upper bound: 0.0497442
time: 6.45 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 50.17 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 50.17
Output dim: 1, lower bound: -0.0497719, upper bound: 0.0497539
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 50.17
Output dim: 1, lower bound: -0.0497638, upper bound: 0.0497657
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 50.17
Output dim: 1, lower bound: -0.0497331, upper bound: 0.0497513
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 50.17
Output dim: 1, lower bound: -0.0497412, upper bound: 0.0497442

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3611989, 0.3612003
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632457, 0.3632454
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183389, 0.3183388
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419262, 0.4419256
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223619, 0.5223625
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168682, 0.4168675
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121763, 0.3121830
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559077, 0.9559075
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1992178, 1.1992185
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419463, 0.6419461

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1014

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497752, upper bound: 0.0497490
time: 114.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497584, upper bound: 0.0497550
time: 7.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3611989, 0.3612003
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632457, 0.3632454
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183389, 0.3183388
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419262, 0.4419256
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223619, 0.5223625
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168683, 0.4168675
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121763, 0.3121830
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559077, 0.9559075
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1992178, 1.1992185
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419463, 0.6419461

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2402

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3430

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497607, upper bound: 0.0497542
time: 5.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497519, upper bound: 0.0497642
time: 25.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3611964, 0.3611987
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632445, 0.3632445
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183399, 0.3183378
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419222, 0.4419168
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223616, 0.5223613
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168614, 0.4168541
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121732, 0.3121572
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559072, 0.9559042
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1992121, 1.1992161
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419449, 0.6419449

Time for backsubstitution: 6.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2634

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496912, upper bound: 0.0497275
time: 83.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0497100, upper bound: 0.0497100
time: 7.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3612004, 0.3611951
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632455, 0.3632447
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183376, 0.3183390
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419162, 0.4419267
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223618, 0.5223620
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168533, 0.4168687
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121638, 0.3121765
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559041, 0.9559078
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1992188, 1.1992114
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419463, 0.6419451

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2553

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3360

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497403, upper bound: 0.0497441
time: 6.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497403, upper bound: 0.0497442
time: 6.33 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 18.84 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.84
Output dim: 1, lower bound: -0.0497752, upper bound: 0.0497490
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.84
Output dim: 1, lower bound: -0.0497584, upper bound: 0.0497550
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.84
Output dim: 1, lower bound: -0.0497607, upper bound: 0.0497542
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.84
Output dim: 1, lower bound: -0.0497519, upper bound: 0.0497642
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.84
Output dim: 1, lower bound: -0.0496912, upper bound: 0.0497275
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 18.84
Output dim: 1, lower bound: -0.0497100, upper bound: 0.0497100
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.84
Output dim: 1, lower bound: -0.0497403, upper bound: 0.0497441
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.84
Output dim: 1, lower bound: -0.0497403, upper bound: 0.0497442

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3611989, 0.3612002
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632456, 0.3632454
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183389, 0.3183388
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419260, 0.4419254
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223619, 0.5223624
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168680, 0.4168673
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121762, 0.3121828
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559072, 0.9559072
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1992171, 1.1992180
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419466, 0.6419463

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2567

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497710, upper bound: 0.0497436
time: 14.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497678, upper bound: 0.0497428
time: 5.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3611989, 0.3612002
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632456, 0.3632454
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183390, 0.3183388
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419260, 0.4419254
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223619, 0.5223624
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168680, 0.4168673
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121762, 0.3121828
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559073, 0.9559072
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1992171, 1.1992180
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419466, 0.6419463

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3346

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3471

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497554, upper bound: 0.0497312
time: 31.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497300, upper bound: 0.0497517
time: 5.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3573113, 0.3574136
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3635139, 0.3634852
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3171499, 0.3171057
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4406958, 0.4406515
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5195187, 0.5195948
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4153720, 0.4153399
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3116341, 0.3116523
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9548529, 0.9548321
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1984339, 1.1984489
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6412872, 0.6412556

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2551

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3498

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497569, upper bound: 0.0497424
time: 5.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497474, upper bound: 0.0497490
time: 6.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3574121, 0.3573127
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3634855, 0.3635138
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3171059, 0.3171497
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4406521, 0.4406952
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5195943, 0.5195192
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4153407, 0.4153712
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3116458, 0.3116406
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9548323, 0.9548526
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1984477, 1.1984351
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6412557, 0.6412871

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2510

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497439, upper bound: 0.0496870
time: 131.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496836, upper bound: 0.0497576
time: 5.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3609097, 0.3609200
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3630950, 0.3630953
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3179697, 0.3179662
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4394108, 0.4393634
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5217140, 0.5216964
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4138449, 0.4137860
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3111655, 0.3111280
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9539982, 0.9539733
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986406, 1.1986609
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6417196, 0.6417232

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2605

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2932

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496922, upper bound: 0.0497271
time: 57.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496908, upper bound: 0.0497271
time: 69.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3612004, 0.3611951
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632455, 0.3632447
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183376, 0.3183390
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419162, 0.4419267
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223618, 0.5223620
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168533, 0.4168687
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121638, 0.3121765
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559041, 0.9559078
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1992188, 1.1992114
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419463, 0.6419451

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2634

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2325

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497376, upper bound: 0.0497369
time: 43.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497315, upper bound: 0.0497411
time: 67.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3612004, 0.3611951
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632455, 0.3632447
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183376, 0.3183390
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419162, 0.4419267
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223618, 0.5223620
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168533, 0.4168687
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121638, 0.3121765
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559041, 0.9559078
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1992188, 1.1992114
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419463, 0.6419451

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2631

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2775

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497357, upper bound: 0.0497413
time: 5.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497355, upper bound: 0.0497416
time: 36.49 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 48.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0497710, upper bound: 0.0497436
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0497678, upper bound: 0.0497428
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0497554, upper bound: 0.0497312
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0497300, upper bound: 0.0497517
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0497569, upper bound: 0.0497424
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0497474, upper bound: 0.0497490
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0497439, upper bound: 0.0496870
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0496836, upper bound: 0.0497576
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0496922, upper bound: 0.0497271
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0496908, upper bound: 0.0497271
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0497376, upper bound: 0.0497369
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0497315, upper bound: 0.0497411
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0497357, upper bound: 0.0497413
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 48.20
Output dim: 1, lower bound: -0.0497355, upper bound: 0.0497416

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3607998, 0.3607989
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632145, 0.3632151
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183115, 0.3183121
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4415735, 0.4415637
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223470, 0.5223461
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4161744, 0.4161689
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3119260, 0.3119328
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9556236, 0.9556242
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1990347, 1.1990337
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419332, 0.6419321

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2635

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3327

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497711, upper bound: 0.0497438
time: 12.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497680, upper bound: 0.0497435
time: 5.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3607975, 0.3608013
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632153, 0.3632143
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183123, 0.3183113
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4415643, 0.4415730
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223455, 0.5223475
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4161696, 0.4161736
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3119261, 0.3119326
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9556246, 0.9556235
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1990323, 1.1990356
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419322, 0.6419331

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3498

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 441

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497675, upper bound: 0.0497400
time: 15.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497678, upper bound: 0.0497435
time: 63.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3611867, 0.3611859
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632417, 0.3632408
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183365, 0.3183385
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4418901, 0.4418922
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5221491, 0.5221541
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4162329, 0.4162622
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3099730, 0.3101444
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9551266, 0.9551431
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1988287, 1.1988311
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6411399, 0.6411939

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 698

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497531, upper bound: 0.0497314
time: 6.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497533, upper bound: 0.0497315
time: 18.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3611844, 0.3611881
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632410, 0.3632416
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183388, 0.3183363
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4418928, 0.4418895
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5221536, 0.5221497
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4162629, 0.4162321
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3101378, 0.3099796
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9551434, 0.9551264
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1988301, 1.1988297
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6411942, 0.6411395

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3312

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497313, upper bound: 0.0497511
time: 5.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497322, upper bound: 0.0497503
time: 5.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3572001, 0.3573121
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3634892, 0.3634441
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3169132, 0.3169038
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4403164, 0.4402417
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5194682, 0.5195371
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4152545, 0.4152082
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3116199, 0.3116377
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9548014, 0.9547739
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1984119, 1.1984277
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6412795, 0.6412475

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3077

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2326

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497553, upper bound: 0.0497352
time: 5.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497508, upper bound: 0.0497401
time: 19.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3572099, 0.3573024
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3634728, 0.3634605
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3169481, 0.3168689
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4402860, 0.4402722
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5194610, 0.5195442
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4152403, 0.4152223
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3116195, 0.3116382
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9547945, 0.9547806
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1984129, 1.1984267
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6412792, 0.6412475

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 777

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 721

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497450, upper bound: 0.0497501
time: 18.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497501, upper bound: 0.0497466
time: 6.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3562117, 0.3560795
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3628371, 0.3628484
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3148862, 0.3149897
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4286512, 0.4290100
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5179256, 0.5178936
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4012963, 0.4016595
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3068241, 0.3069662
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9490724, 0.9492466
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1969707, 1.1969228
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6413552, 0.6413937

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3446

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1043

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497451, upper bound: 0.0496974
time: 5.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497452, upper bound: 0.0496870
time: 151.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3561788, 0.3561124
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3628202, 0.3628656
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3149458, 0.3149301
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4289669, 0.4286942
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5179685, 0.5178505
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4016290, 0.4013268
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3069713, 0.3068190
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9492263, 0.9490929
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1969357, 1.1969581
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6413623, 0.6413865

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2660

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3360

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496836, upper bound: 0.0497570
time: 6.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496836, upper bound: 0.0497576
time: 5.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3609072, 0.3609174
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3630896, 0.3630896
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3179678, 0.3179643
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4393968, 0.4393489
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5217046, 0.5216868
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4138254, 0.4137665
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3111674, 0.3111301
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9540024, 0.9539775
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986365, 1.1986568
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6417080, 0.6417115

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 1028

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496872, upper bound: 0.0496879
time: 6.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496523, upper bound: 0.0497226
time: 35.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3609073, 0.3609174
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3630893, 0.3630900
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3179677, 0.3179643
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4393963, 0.4393493
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5217044, 0.5216870
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4138254, 0.4137665
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3111675, 0.3111300
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9540023, 0.9539777
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1986365, 1.1986568
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6417075, 0.6417115

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3360
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 1026

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 698

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496902, upper bound: 0.0497272
time: 14.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496914, upper bound: 0.0497278
time: 23.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3611454, 0.3611004
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632450, 0.3632438
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183317, 0.3183359
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419116, 0.4419212
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223504, 0.5223336
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168279, 0.4168485
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121564, 0.3121709
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559020, 0.9559058
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1991706, 1.1991260
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419325, 0.6419358

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3498

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2503

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497362, upper bound: 0.0497283
time: 47.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497304, upper bound: 0.0497341
time: 9.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.6079993, -0.7557846, -1.6079993, -0.7557846, -0.3611059, 0.3611400
1: -0.1078633, 0.4254003, -0.1078633, 0.4254003, -0.3632447, 0.3632441
2: -1.6562660, -1.0956314, -1.6562660, -1.0956314, -0.3183344, 0.3183331
3: -4.2293043, -2.7018771, -4.2293043, -2.7018771, -0.4419108, 0.4419221
4: -1.7455292, -0.8654980, -1.7455292, -0.8654980, -0.5223333, 0.5223507
5: -4.2149248, -2.4606175, -4.2149248, -2.4606175, -0.4168331, 0.4168434
6: -5.7988276, -4.2572432, -5.7988276, -4.2572432, -0.3121582, 0.3121691
7: -0.9004182, 0.6099422, -0.9004182, 0.6099422, -0.9559023, 0.9559056
8: -2.2988765, -0.6395774, -2.2988765, -0.6395774, -1.1991334, 1.1991632
9: -0.8526962, 0.0945601, -0.8526962, 0.0945601, -0.6419370, 0.6419315

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 1028
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 1043
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 1026
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1015
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 1030
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 1002
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 1014
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 1016
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 1029
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2326

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497328, upper bound: 0.0497277
time: 17.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497184, upper bound: 0.0497409
time: 153.19 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 177.57 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497711, upper bound: 0.0497438
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497680, upper bound: 0.0497435
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497675, upper bound: 0.0497400
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497678, upper bound: 0.0497435
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497531, upper bound: 0.0497314
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497533, upper bound: 0.0497315
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497313, upper bound: 0.0497511
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497322, upper bound: 0.0497503
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497553, upper bound: 0.0497352
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497508, upper bound: 0.0497401
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497450, upper bound: 0.0497501
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497501, upper bound: 0.0497466
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497451, upper bound: 0.0496974
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497452, upper bound: 0.0496870
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0496836, upper bound: 0.0497570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0496836, upper bound: 0.0497576
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0496872, upper bound: 0.0496879
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0496523, upper bound: 0.0497226
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0496902, upper bound: 0.0497272
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0496914, upper bound: 0.0497278
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497362, upper bound: 0.0497283
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497304, upper bound: 0.0497341
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497328, upper bound: 0.0497277
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 177.57
Output dim: 1, lower bound: -0.0497184, upper bound: 0.0497409
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 177.57
Output dim: 1, lower bound: -0.0497357, upper bound: 0.0497413
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 177.57
Output dim: 1, lower bound: -0.0497355, upper bound: 0.0497416

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 75.58 + 1748.34 = 1823.93 seconds
