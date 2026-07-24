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
execution time: IAR + RelationalAnalysis = 7.82 + 65.64 = 73.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0497747, upper bound: 0.0497754

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3401

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497720, upper bound: 0.0496591
time: 5.34 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497740, upper bound: 0.0497755
time: 33.14 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 38.55 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 38.55
Output dim: 1, lower bound: -0.0497720, upper bound: 0.0496591
NS_A2, status: Status.UNKNOWN, split count: 1, time: 38.55
Output dim: 1, lower bound: -0.0497740, upper bound: 0.0497755

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1.6043134, -0.7581333, -1.6046966, -0.7558301, -0.3578890, 0.3558971
1: -0.1061063, 0.4228671, -0.1078095, 0.4231091, -0.3593416, 0.3607653
2: -1.6561130, -1.0956798, -1.6561252, -1.0957062, -0.3180351, 0.3178349
3: -4.2277255, -2.7043598, -4.2292776, -2.7041039, -0.4385099, 0.4397832
4: -1.7444637, -0.8659035, -1.7446984, -0.8655329, -0.5214396, 0.5214454
5: -4.2124610, -2.4640563, -4.2149210, -2.4637191, -0.4114664, 0.4135590
6: -5.7983632, -4.2574534, -5.7984147, -4.2572489, -0.3117115, 0.3115383
7: -0.8994699, 0.6094642, -0.9003837, 0.6096473, -0.9546828, 0.9552257
8: -2.2955861, -0.6417215, -2.2959306, -0.6395924, -1.1959321, 1.1941376
9: -0.8512660, 0.0922484, -0.8526452, 0.0924451, -0.6388222, 0.6397623

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3546
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2093
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 312
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 3230
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 3471
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2952
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3178
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 3328
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3360
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3369
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2388

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0497012, upper bound: 0.0496317
time: 17.25 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497466, upper bound: 0.0496317
time: 6.51 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.6079822, -0.7557847, -1.6079841, -0.7557848, -0.3564688, 0.3608925
1: -0.1078631, 0.4253840, -0.1078632, 0.4253855, -0.3632208, 0.3594011
2: -1.6562414, -1.0956316, -1.6562437, -1.0956315, -0.3182511, 0.3184024
3: -4.2293043, -2.7018902, -4.2293043, -2.7018881, -0.4416246, 0.4390112
4: -1.7455003, -0.8654982, -1.7455013, -0.8654984, -0.5221136, 0.5219685
5: -4.2149243, -2.4606342, -4.2149243, -2.4606328, -0.4168124, 0.4115496
6: -5.7987866, -4.2572436, -5.7987905, -4.2572432, -0.3117532, 0.3120483
7: -0.9004103, 0.6099229, -0.9004103, 0.6099244, -0.9554257, 0.9562496
8: -2.2988620, -0.6395786, -2.2988636, -0.6395786, -1.1950421, 1.1992135
9: -0.8526957, 0.0943463, -0.8526955, 0.0943675, -0.6415169, 0.6393316

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3546
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2093
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 312
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 3230
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 3471
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2952
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3178
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 3328
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3360
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3369
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2388

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496987, upper bound: 0.0497458
time: 6.74 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497461, upper bound: 0.0497468
time: 87.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 100.27 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 100.27
Output dim: 1, lower bound: -0.0497012, upper bound: 0.0496317
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 100.27
Output dim: 1, lower bound: -0.0497466, upper bound: 0.0496317
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 100.27
Output dim: 1, lower bound: -0.0496987, upper bound: 0.0497458
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 100.27
Output dim: 1, lower bound: -0.0497461, upper bound: 0.0497468

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1.6042879, -0.7591511, -1.6046678, -0.7569564, -0.3572058, 0.3551062
1: -0.1060617, 0.4224483, -0.1077600, 0.4226508, -0.3581702, 0.3603559
2: -1.6555378, -1.0956897, -1.6554861, -1.0957174, -0.3178156, 0.3157932
3: -4.2250357, -2.7043612, -4.2262845, -2.7041066, -0.4383693, 0.4298711
4: -1.7440619, -0.8659106, -1.7442499, -0.8655411, -0.5212415, 0.5198102
5: -4.2105346, -2.4640579, -4.2128048, -2.4637222, -0.4112610, 0.3997559
6: -5.7966547, -4.2574615, -5.7965121, -4.2572570, -0.3115982, 0.3072500
7: -0.8979897, 0.6093953, -0.8987399, 0.6095703, -0.9539811, 0.9470592
8: -2.2955320, -0.6436322, -2.2958710, -0.6416872, -1.1923413, 1.1924129
9: -0.8512033, 0.0912542, -0.8525752, 0.0913727, -0.6377804, 0.6388574

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2635

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496600, upper bound: 0.0496222
time: 62.08 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497367, upper bound: 0.0496223
time: 6.43 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.6079037, -0.7563635, -1.6093460, -0.7564228, -0.3554915, 0.3609697
1: -0.1077472, 0.4237154, -0.1075354, 0.4235374, -0.3612213, 0.3572819
2: -1.6551893, -1.0956498, -1.6550786, -1.0952080, -0.3162952, 0.3168336
3: -4.2258968, -2.7018926, -4.2255831, -2.6989098, -0.4319394, 0.4321644
4: -1.7444588, -0.8655149, -1.7443492, -0.8654351, -0.5205317, 0.5206240
5: -4.2109680, -2.4606397, -4.2106018, -2.4584489, -0.4035672, 0.4024549
6: -5.7979517, -4.2572637, -5.7978735, -4.2545857, -0.3076006, 0.3089209
7: -0.8946105, 0.6098082, -0.8940455, 0.6090252, -0.9464914, 0.9492316
8: -2.2987168, -0.6452303, -2.2995763, -0.6456892, -1.1875143, 1.1914592
9: -0.8524776, 0.0912945, -0.8522985, 0.0909941, -0.6379743, 0.6358349

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2635

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496114, upper bound: 0.0497361
time: 16.23 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496883, upper bound: 0.0497362
time: 7.35 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1.6079559, -0.7568028, -1.6079550, -0.7569112, -0.3557844, 0.3601018
1: -0.1078179, 0.4249650, -0.1078134, 0.4249268, -0.3620485, 0.3589920
2: -1.6556667, -1.0956416, -1.6556051, -1.0956426, -0.3180316, 0.3163604
3: -4.2266140, -2.7018921, -4.2263117, -2.7018912, -0.4414840, 0.4290940
4: -1.7450984, -0.8655058, -1.7450533, -0.8655064, -0.5219156, 0.5203329
5: -4.2129984, -2.4606361, -4.2128086, -2.4606347, -0.4166057, 0.3977471
6: -5.7970791, -4.2572517, -5.7968874, -4.2572522, -0.3116398, 0.3077586
7: -0.8989307, 0.6098536, -0.8987678, 0.6098473, -0.9547238, 0.9480848
8: -2.2988079, -0.6414895, -2.2988050, -0.6416724, -1.1914520, 1.1974897
9: -0.8526312, 0.0933514, -0.8526251, 0.0932956, -0.6404746, 0.6384275

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2635

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496591, upper bound: 0.0497374
time: 20.11 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497369, upper bound: 0.0497377
time: 20.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 46.98 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 46.98
Output dim: 1, lower bound: -0.0496600, upper bound: 0.0496222
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 46.98
Output dim: 1, lower bound: -0.0497367, upper bound: 0.0496223
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 46.98
Output dim: 1, lower bound: -0.0496114, upper bound: 0.0497361
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 46.98
Output dim: 1, lower bound: -0.0496883, upper bound: 0.0497362
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 46.98
Output dim: 1, lower bound: -0.0496591, upper bound: 0.0497374
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 46.98
Output dim: 1, lower bound: -0.0497369, upper bound: 0.0497377

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.6041894, -0.7591553, -1.6045806, -0.7569601, -0.3558368, 0.3550225
1: -0.1059233, 0.4224483, -0.1076380, 0.4226508, -0.3574197, 0.3602320
2: -1.6555370, -1.0958261, -1.6554856, -1.0958376, -0.3177336, 0.3144011
3: -4.2250347, -2.7046266, -4.2262821, -2.7043414, -0.4381415, 0.4232835
4: -1.7440540, -0.8662067, -1.7442430, -0.8658025, -0.5210918, 0.5178716
5: -4.2105350, -2.4644036, -4.2128057, -2.4640265, -0.4111100, 0.3909950
6: -5.7966499, -4.2575483, -5.7965083, -4.2573338, -0.3112329, 0.3061938
7: -0.8979888, 0.6090487, -0.8987387, 0.6092644, -0.9538175, 0.9400544
8: -2.2949231, -0.6436322, -2.2953348, -0.6416876, -1.1897416, 1.1918592
9: -0.8508974, 0.0912540, -0.8523052, 0.0913723, -0.6369972, 0.6385748

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3546
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2093
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 312
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 3230
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 3471
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2952
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3178
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 3328
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3360
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3369
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2178

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496553, upper bound: 0.0496148
time: 44.78 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497302, upper bound: 0.0496150
time: 40.08 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.6063529, -0.7569278, -1.6079962, -0.7565308, -0.3537416, 0.3588520
1: -0.1046540, 0.4228373, -0.1048260, 0.4235374, -0.3581688, 0.3537191
2: -1.6546388, -1.0980618, -1.6550715, -1.0972958, -0.3131373, 0.3142899
3: -4.2245998, -2.7065377, -4.2255292, -2.7031026, -0.4251044, 0.4271031
4: -1.7429880, -0.8701273, -1.7441223, -0.8695204, -0.5145066, 0.5157319
5: -4.2093344, -2.4666612, -4.2105989, -2.4638817, -0.3949609, 0.3959951
6: -5.7974606, -4.2585678, -5.7977581, -4.2557530, -0.3057507, 0.3075159
7: -0.8924625, 0.6035802, -0.8940394, 0.6034596, -0.9378363, 0.9426682
8: -2.2861390, -0.6487818, -2.2882941, -0.6456902, -1.1749949, 1.1766934
9: -0.8460357, 0.0894723, -0.8465927, 0.0909925, -0.6316148, 0.6283147

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3546
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2093
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 312
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 3230
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 3471
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2952
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3178
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 3328
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3360
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3369
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2178

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0495326, upper bound: 0.0497301
time: 31.23 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496084, upper bound: 0.0497309
time: 49.47 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.6078050, -0.7563676, -1.6092591, -0.7564267, -0.3541221, 0.3608858
1: -0.1076092, 0.4237154, -0.1074136, 0.4235374, -0.3604707, 0.3571581
2: -1.6551889, -1.0957863, -1.6550779, -1.0953283, -0.3162133, 0.3154416
3: -4.2258949, -2.7021573, -4.2255812, -2.6991444, -0.4317117, 0.4255776
4: -1.7444506, -0.8658112, -1.7443414, -0.8656966, -0.5203817, 0.5186850
5: -4.2109675, -2.4609859, -4.2106004, -2.4587543, -0.4034164, 0.3936935
6: -5.7979469, -4.2573509, -5.7978687, -4.2546630, -0.3072354, 0.3078642
7: -0.8946102, 0.6094623, -0.8940448, 0.6087198, -0.9463277, 0.9422271
8: -2.2981079, -0.6452305, -2.2990398, -0.6456900, -1.1849155, 1.1909056
9: -0.8521724, 0.0912945, -0.8520290, 0.0909941, -0.6371917, 0.6355523

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3546
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2093
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 312
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 3230
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 3471
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2952
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3178
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 3328
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3360
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3369
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2178

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496104, upper bound: 0.0497309
time: 127.16 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496848, upper bound: 0.0497314
time: 18.75 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.6064057, -0.7573673, -1.6066065, -0.7570190, -0.3540354, 0.3579852
1: -0.1047251, 0.4240868, -0.1051040, 0.4249268, -0.3589964, 0.3554294
2: -1.6551157, -1.0980539, -1.6555984, -1.0977305, -0.3148743, 0.3138166
3: -4.2253170, -2.7065387, -4.2262568, -2.7060843, -0.4346488, 0.4240327
4: -1.7436275, -0.8701181, -1.7448266, -0.8695914, -0.5158903, 0.5154412
5: -4.2113652, -2.4666572, -4.2128057, -2.4660664, -0.4079999, 0.3912877
6: -5.7965879, -4.2585554, -5.7967720, -4.2584200, -0.3097898, 0.3063536
7: -0.8967819, 0.6036251, -0.8987622, 0.6042826, -0.9460695, 0.9415202
8: -2.2862339, -0.6450408, -2.2875221, -0.6416733, -1.1789317, 1.1827226
9: -0.8461913, 0.0915294, -0.8469211, 0.0932941, -0.6341153, 0.6309063

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3546
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2093
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 312
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 3230
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 3471
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2952
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3178
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 3328
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3360
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3369
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2178

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0495797, upper bound: 0.0497311
time: 47.86 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496539, upper bound: 0.0497319
time: 7.82 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.6078568, -0.7568071, -1.6078680, -0.7569149, -0.3544155, 0.3600179
1: -0.1076794, 0.4249650, -0.1076909, 0.4249268, -0.3612979, 0.3588682
2: -1.6556656, -1.0957779, -1.6556046, -1.0957628, -0.3179498, 0.3149683
3: -4.2266121, -2.7021580, -4.2263103, -2.7021255, -0.4412560, 0.4225070
4: -1.7450900, -0.8658023, -1.7450458, -0.8657678, -0.5217656, 0.5183942
5: -4.2129984, -2.4609818, -4.2128081, -2.4609399, -0.4164547, 0.3889862
6: -5.7970743, -4.2573395, -5.7968836, -4.2573295, -0.3112746, 0.3067015
7: -0.8989305, 0.6095068, -0.8987672, 0.6095417, -0.9545605, 0.9410803
8: -2.2981994, -0.6414895, -2.2982674, -0.6416731, -1.1888528, 1.1969361
9: -0.8523262, 0.0933518, -0.8523557, 0.0932958, -0.6396917, 0.6381452

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3546
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2093
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 312
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 3230
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 3471
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2952
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3178
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 3328
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3360
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3369
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2178

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0496549, upper bound: 0.0497306
time: 34.60 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0497298, upper bound: 0.0497304
time: 62.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 103.23 seconds
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 103.23
Output dim: 1, lower bound: -0.0496553, upper bound: 0.0496148
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 103.23
Output dim: 1, lower bound: -0.0497302, upper bound: 0.0496150
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 103.23
Output dim: 1, lower bound: -0.0495326, upper bound: 0.0497301
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 103.23
Output dim: 1, lower bound: -0.0496084, upper bound: 0.0497309
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 103.23
Output dim: 1, lower bound: -0.0496104, upper bound: 0.0497309
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 103.23
Output dim: 1, lower bound: -0.0496848, upper bound: 0.0497314
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 103.23
Output dim: 1, lower bound: -0.0495797, upper bound: 0.0497311
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 103.23
Output dim: 1, lower bound: -0.0496539, upper bound: 0.0497319
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 103.23
Output dim: 1, lower bound: -0.0496549, upper bound: 0.0497306
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 103.23
Output dim: 1, lower bound: -0.0497298, upper bound: 0.0497304

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.6041845, -0.7592359, -1.6045754, -0.7570502, -0.3551598, 0.3549322
1: -0.1059089, 0.4223450, -0.1076213, 0.4225354, -0.3566742, 0.3601217
2: -1.6554480, -1.0958407, -1.6553862, -1.0958539, -0.3176318, 0.3133821
3: -4.2248163, -2.7046275, -4.2260399, -2.7043414, -0.4377984, 0.4194165
4: -1.7438600, -0.8662142, -1.7440352, -0.8658109, -0.5209512, 0.5169699
5: -4.2103353, -2.4644041, -4.2125840, -2.4640274, -0.4110378, 0.3857949
6: -5.7965117, -4.2575512, -5.7963533, -4.2573357, -0.3107836, 0.3036526
7: -0.8977740, 0.6090264, -0.8984989, 0.6092389, -0.9535232, 0.9372404
8: -2.2949059, -0.6441526, -2.2953153, -0.6422672, -1.1879587, 1.1914880
9: -0.8508854, 0.0910401, -0.8522922, 0.0911369, -0.6362197, 0.6383878

Time for backsubstitution: 6.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2606

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496784, upper bound: 0.0495865
time: 57.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496998, upper bound: 0.0495859
time: 34.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.6062591, -0.7577776, -1.6076461, -0.7575105, -0.3526582, 0.3576491
1: -0.1044207, 0.4209177, -0.1038898, 0.4214243, -0.3555878, 0.3502784
2: -1.6527731, -1.0983326, -1.6530335, -1.0983894, -0.3104136, 0.3122195
3: -4.2224307, -2.7065377, -4.2230973, -2.7040024, -0.4213598, 0.4244704
4: -1.7415702, -0.8702606, -1.7425443, -0.8699554, -0.5123288, 0.5138999
5: -4.2059422, -2.4666677, -4.2068343, -2.4652791, -0.3897134, 0.3922398
6: -5.7960649, -4.2585969, -5.7961674, -4.2563038, -0.3031958, 0.3057876
7: -0.8886601, 0.6033819, -0.8898402, 0.6020095, -0.9329014, 0.9385134
8: -2.2858458, -0.6562943, -2.2846675, -0.6539087, -1.1662960, 1.1649175
9: -0.8457720, 0.0861640, -0.8452312, 0.0873306, -0.6271129, 0.6221656

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2606

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0494805, upper bound: 0.0497010
time: 7.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0495048, upper bound: 0.0497015
time: 30.15 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.6063478, -0.7570084, -1.6079915, -0.7566206, -0.3530647, 0.3587613
1: -0.1046391, 0.4227338, -0.1048090, 0.4234222, -0.3574232, 0.3536087
2: -1.6545498, -1.0980766, -1.6549733, -1.0973120, -0.3130360, 0.3132712
3: -4.2243819, -2.7065372, -4.2252846, -2.7031031, -0.4247570, 0.4232283
4: -1.7427934, -0.8701345, -1.7439150, -0.8695284, -0.5143663, 0.5148310
5: -4.2091298, -2.4666619, -4.2103696, -2.4638813, -0.3948846, 0.3907952
6: -5.7973228, -4.2585707, -5.7976036, -4.2557554, -0.3052994, 0.3049552
7: -0.8922451, 0.6035569, -0.8937972, 0.6034334, -0.9375468, 0.9398537
8: -2.2861221, -0.6493156, -2.2882752, -0.6462860, -1.1732044, 1.1763129
9: -0.8460224, 0.0892582, -0.8465781, 0.0907569, -0.6308321, 0.6281273

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2606

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0495567, upper bound: 0.0497019
time: 6.74 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0495797, upper bound: 0.0497028
time: 13.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.6077120, -0.7572201, -1.6089091, -0.7574074, -0.3530391, 0.3596803
1: -0.1073775, 0.4217958, -0.1064787, 0.4214243, -0.3578911, 0.3537176
2: -1.6533237, -1.0960546, -1.6530397, -1.0964215, -0.3134895, 0.3133716
3: -4.2237229, -2.7021573, -4.2231493, -2.7000439, -0.4279648, 0.4229464
4: -1.7430334, -0.8659436, -1.7427645, -0.8661305, -0.5182038, 0.5168532
5: -4.2075744, -2.4609931, -4.2068372, -2.4601519, -0.3981692, 0.3899379
6: -5.7965488, -4.2573786, -5.7962780, -4.2552137, -0.3046753, 0.3061385
7: -0.8908069, 0.6092639, -0.8898449, 0.6072704, -0.9413921, 0.9380708
8: -2.2978182, -0.6527424, -2.2954161, -0.6539078, -1.1762176, 1.1791315
9: -0.8519124, 0.0879855, -0.8506703, 0.0873315, -0.6326895, 0.6294041

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2606

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0495585, upper bound: 0.0497021
time: 11.20 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0495813, upper bound: 0.0497014
time: 55.36 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.6077998, -0.7564483, -1.6092536, -0.7565164, -0.3534451, 0.3607947
1: -0.1075946, 0.4236119, -0.1073970, 0.4234222, -0.3597253, 0.3570477
2: -1.6551001, -1.0957999, -1.6549793, -1.0953442, -0.3161120, 0.3144226
3: -4.2256765, -2.7021568, -4.2253366, -2.6991446, -0.4313642, 0.4217033
4: -1.7442572, -0.8658183, -1.7441343, -0.8657044, -0.5202411, 0.5177834
5: -4.2107625, -2.4609866, -4.2103729, -2.4587545, -0.4033400, 0.3884941
6: -5.7978096, -4.2573538, -5.7977138, -4.2546654, -0.3067837, 0.3053044
7: -0.8943928, 0.6094389, -0.8938028, 0.6086940, -0.9460384, 0.9394121
8: -2.2980914, -0.6457639, -2.2990201, -0.6462848, -1.1831245, 1.1905251
9: -0.8521597, 0.0910804, -0.8520147, 0.0907583, -0.6364086, 0.6353660

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2606

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496330, upper bound: 0.0497021
time: 16.92 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496554, upper bound: 0.0497013
time: 17.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.6063126, -0.7582167, -1.6062527, -0.7579989, -0.3529502, 0.3567775
1: -0.1044925, 0.4221672, -0.1041684, 0.4228135, -0.3564160, 0.3519893
2: -1.6532518, -1.0983243, -1.6535614, -1.0988240, -0.3121502, 0.3117478
3: -4.2231483, -2.7065392, -4.2238259, -2.7069838, -0.4308873, 0.4213990
4: -1.7422104, -0.8702515, -1.7432488, -0.8700264, -0.5137117, 0.5136096
5: -4.2079697, -2.4666650, -4.2090373, -2.4674811, -0.4027510, 0.3875328
6: -5.7951918, -4.2585835, -5.7951822, -4.2589703, -0.3072128, 0.3046220
7: -0.8929753, 0.6034271, -0.8945578, 0.6028332, -0.9411309, 0.9373617
8: -2.2859404, -0.6525593, -2.2838793, -0.6498997, -1.1702232, 1.1709194
9: -0.8459294, 0.0882208, -0.8455603, 0.0896313, -0.6296029, 0.6247460

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2606

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0495262, upper bound: 0.0497000
time: 47.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0495482, upper bound: 0.0497004
time: 97.47 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.6064008, -0.7574477, -1.6066010, -0.7571090, -0.3533585, 0.3578952
1: -0.1047105, 0.4239836, -0.1050874, 0.4248116, -0.3582509, 0.3553191
2: -1.6550269, -1.0980685, -1.6554995, -1.0977467, -0.3147724, 0.3127978
3: -4.2251005, -2.7065387, -4.2260137, -2.7060840, -0.4343062, 0.4201656
4: -1.7434335, -0.8701254, -1.7446189, -0.8695998, -0.5157501, 0.5145400
5: -4.2111659, -2.4666572, -4.2125831, -2.4660668, -0.4079272, 0.3860875
6: -5.7964492, -4.2585578, -5.7966180, -4.2584214, -0.3093409, 0.3038100
7: -0.8965652, 0.6036024, -0.8985204, 0.6042560, -0.9457755, 0.9387076
8: -2.2862160, -0.6455617, -2.2875028, -0.6422536, -1.1771500, 1.1823525
9: -0.8461783, 0.0913157, -0.8469063, 0.0930579, -0.6333377, 0.6307197

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2606

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496028, upper bound: 0.0497026
time: 90.29 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496239, upper bound: 0.0497018
time: 6.97 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.6077652, -0.7576589, -1.6075145, -0.7578956, -0.3533312, 0.3588078
1: -0.1074489, 0.4230455, -0.1067569, 0.4228135, -0.3587187, 0.3554286
2: -1.6538022, -1.0960462, -1.6535678, -1.0968567, -0.3152257, 0.3128999
3: -4.2244406, -2.7021577, -4.2238784, -2.7030263, -0.4374925, 0.4198748
4: -1.7436733, -0.8659347, -1.7434688, -0.8662020, -0.5195872, 0.5165631
5: -4.2096019, -2.4609895, -4.2090416, -2.4623542, -0.4112063, 0.3852308
6: -5.7956767, -4.2573667, -5.7952924, -4.2578797, -0.3086927, 0.3049728
7: -0.8951226, 0.6093092, -0.8945630, 0.6080942, -0.9496207, 0.9369202
8: -2.2979116, -0.6490078, -2.2946274, -0.6498985, -1.1801441, 1.1851320
9: -0.8520674, 0.0900426, -0.8509973, 0.0896320, -0.6351798, 0.6319849

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2606

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496033, upper bound: 0.0497005
time: 66.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496268, upper bound: 0.0497015
time: 72.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.6078522, -0.7568874, -1.6078627, -0.7570050, -0.3537383, 0.3599275
1: -0.1076651, 0.4248619, -0.1076748, 0.4248116, -0.3605525, 0.3587579
2: -1.6555767, -1.0957924, -1.6555057, -1.0957791, -0.3178481, 0.3139494
3: -4.2263942, -2.7021577, -4.2260671, -2.7021260, -0.4409132, 0.4186400
4: -1.7448971, -0.8658091, -1.7448387, -0.8657761, -0.5216252, 0.5174925
5: -4.2127995, -2.4609823, -4.2125874, -2.4609396, -0.4163823, 0.3837863
6: -5.7969351, -4.2573409, -5.7967291, -4.2573314, -0.3108254, 0.3041593
7: -0.8987151, 0.6094843, -0.8985269, 0.6095159, -0.9542662, 0.9382669
8: -2.2981834, -0.6420097, -2.2982488, -0.6422541, -1.1870694, 1.1965644
9: -0.8523138, 0.0931380, -0.8523412, 0.0930598, -0.6389138, 0.6379580

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 3328
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3360
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2606

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0496794, upper bound: 0.0497015
time: 90.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0497002, upper bound: 0.0497021
time: 5.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 102.76 seconds
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0496784, upper bound: 0.0495865
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0496998, upper bound: 0.0495859
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0494805, upper bound: 0.0497010
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0495048, upper bound: 0.0497015
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0495567, upper bound: 0.0497019
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0495797, upper bound: 0.0497028
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0495585, upper bound: 0.0497021
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0495813, upper bound: 0.0497014
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0496330, upper bound: 0.0497021
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0496554, upper bound: 0.0497013
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0495262, upper bound: 0.0497000
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0495482, upper bound: 0.0497004
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0496028, upper bound: 0.0497026
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0496239, upper bound: 0.0497018
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0496033, upper bound: 0.0497005
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0496268, upper bound: 0.0497015
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0496794, upper bound: 0.0497015
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 102.76
Output dim: 1, lower bound: -0.0497002, upper bound: 0.0497021

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 73.46 + 1599.15 = 1672.61 seconds
