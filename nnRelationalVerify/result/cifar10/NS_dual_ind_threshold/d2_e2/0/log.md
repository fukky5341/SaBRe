## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.051055793099999996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930596, 0.7930597)
1: (-0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1681775, 0.1681775)
2: (-4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2614994, 0.2614994)
3: (-3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5348346, 0.5348346)
4: (-4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3903269, 0.3903269)
5: (-3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4016791, 0.4016791)
6: (-4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7665814, 0.7665814)
7: (-4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5072817, 0.5072817)
8: (0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1570090, 0.1570090)
9: (-1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442252, 0.4442252)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.01 + 22.37 = 30.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0511057, upper bound: 0.0511097

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 399
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3513
type: A, layer: 1, pos: 3402
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 360
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 3390
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3156
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3524
type: A, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 399

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511036, upper bound: 0.0507287
time: 102.46 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511033, upper bound: 0.0511040
time: 60.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 162.99 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 162.99
Output dim: 8, lower bound: -0.0511036, upper bound: 0.0507287
NS_A2, status: Status.UNKNOWN, split count: 1, time: 162.99
Output dim: 8, lower bound: -0.0511033, upper bound: 0.0511040

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.6667159, 0.1476985, -0.6670530, 0.1477039, -0.7916260, 0.7919783
1: -0.2384374, 0.2705322, -0.2387412, 0.2706004, -0.1668190, 0.1670388
2: -4.1239719, -3.1941047, -4.1239080, -3.1937814, -0.2591411, 0.2590792
3: -3.7640586, -2.3567805, -3.7648244, -2.3562517, -0.5323491, 0.5302113
4: -4.9857321, -3.6337786, -4.9851475, -3.6337214, -0.3891312, 0.3890899
5: -3.8385692, -2.4884176, -3.8387215, -2.4881985, -0.4004213, 0.4002511
6: -4.0584402, -2.4213552, -4.0604439, -2.4191349, -0.7588434, 0.7561520
7: -4.2665272, -2.5552328, -4.2673616, -2.5546675, -0.5048142, 0.5059131
8: 0.7726299, 1.1991411, 0.7713978, 1.2001960, -0.1522282, 0.1523889
9: -1.2302778, -0.5058681, -1.2316422, -0.5043064, -0.4388454, 0.4385665

Time for backsubstitution: 6.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 390
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3513
type: B, layer: 1, pos: 3402
type: B, layer: 1, pos: 360
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 455
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 3390
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 3156
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 389
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 1113
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3524
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 390

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0508960, upper bound: 0.0506993
time: 116.91 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511038, upper bound: 0.0507268
time: 120.35 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.6679797, 0.1477194, -0.6679841, 0.1477199, -0.7928195, 0.7929376
1: -0.2396378, 0.2706143, -0.2396388, 0.2706146, -0.1677685, 0.1681768
2: -4.1239195, -3.1924152, -4.1239200, -3.1924040, -0.2614904, 0.2589575
3: -3.7660825, -2.3556304, -3.7660880, -2.3556294, -0.5343803, 0.5368215
4: -4.9853501, -3.6337881, -4.9853511, -3.6337218, -0.3903208, 0.3897216
5: -3.8390460, -2.4877944, -3.8390503, -2.4877887, -0.4016746, 0.4016760
6: -4.0661411, -2.4190261, -4.0661449, -2.4190249, -0.7660956, 0.7688075
7: -4.2699547, -2.5546298, -4.2699614, -2.5546296, -0.5088197, 0.5064740
8: 0.7713039, 1.2036055, 0.7713026, 1.2036071, -0.1569995, 0.1520027
9: -1.2356483, -0.5042364, -1.2356493, -0.5042360, -0.4415621, 0.4442213

Time for backsubstitution: 6.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 390
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3513
type: B, layer: 1, pos: 3402
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 360
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 455
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 3390
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 3156
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 389
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 1113
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3524
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 390

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0508955, upper bound: 0.0510858
time: 7.01 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511038, upper bound: 0.0511046
time: 52.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 66.21 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 66.21
Output dim: 8, lower bound: -0.0508960, upper bound: 0.0506993
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 66.21
Output dim: 8, lower bound: -0.0511038, upper bound: 0.0507268
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 66.21
Output dim: 8, lower bound: -0.0508955, upper bound: 0.0510858
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 66.21
Output dim: 8, lower bound: -0.0511038, upper bound: 0.0511046

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.6667087, 0.1476985, -0.6670426, 0.1477038, -0.7916183, 0.7895088
1: -0.2384372, 0.2703116, -0.2387410, 0.2702812, -0.1665553, 0.1670172
2: -4.1239719, -3.1941073, -4.1239076, -3.1937847, -0.2614554, 0.2583487
3: -3.7640576, -2.3567829, -3.7648234, -2.3562546, -0.5299214, 0.5301615
4: -4.9857316, -3.6337857, -4.9851470, -3.6337323, -0.3875245, 0.3886405
5: -3.8385682, -2.4884191, -3.8387213, -2.4882002, -0.3996877, 0.3999021
6: -4.0584407, -2.4213600, -4.0604439, -2.4191415, -0.7568451, 0.7559231
7: -4.2665257, -2.5552452, -4.2673607, -2.5546854, -0.4912414, 0.5058895
8: 0.7726300, 1.1991365, 0.7713980, 1.2001894, -0.1484580, 0.1523832
9: -1.2302773, -0.5061218, -1.2316417, -0.5046480, -0.4383184, 0.4410700

Time for backsubstitution: 6.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3513
type: A, layer: 1, pos: 3402
type: A, layer: 1, pos: 360
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 3390
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3156
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3524
type: A, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3528

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0509879, upper bound: 0.0507255
time: 7.09 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511022, upper bound: 0.0507295
time: 6.04 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.6579057, 0.1472057, -0.6545035, 0.1444748, -0.7793657, 0.7788235
1: -0.2387245, 0.2704628, -0.2395929, 0.2704577, -0.1661374, 0.1654878
2: -4.1237698, -3.1955585, -4.1234226, -3.1965694, -0.2580743, 0.2570679
3: -3.7658858, -2.3587482, -3.7647939, -2.3601465, -0.5305383, 0.5329975
4: -4.9849191, -3.6380668, -4.9836693, -3.6395075, -0.3850234, 0.3846432
5: -3.8390157, -2.4899623, -3.8381162, -2.4912362, -0.3990271, 0.4000966
6: -4.0660028, -2.4234488, -4.0653229, -2.4246581, -0.7614871, 0.7652811
7: -4.2688999, -2.5656288, -4.2661753, -2.5684001, -0.4949157, 0.4925084
8: 0.7715902, 1.2007635, 0.7725883, 1.1995641, -0.1529727, 0.1477509
9: -1.2345634, -0.5054103, -1.2350194, -0.5054407, -0.4372244, 0.4382060

Time for backsubstitution: 6.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3513
type: A, layer: 1, pos: 3402
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 360
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3390
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3156
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3524
type: A, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3528

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0507817, upper bound: 0.0510762
time: 152.11 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0508939, upper bound: 0.0510791
time: 68.66 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.6679721, 0.1477193, -0.6679736, 0.1477197, -0.7928118, 0.7904682
1: -0.2396376, 0.2703936, -0.2396387, 0.2702952, -0.1675022, 0.1681552
2: -4.1239200, -3.1924176, -4.1239195, -3.1924081, -0.2638046, 0.2582271
3: -3.7660823, -2.3556328, -3.7660871, -2.3556323, -0.5320030, 0.5367718
4: -4.9853492, -3.6337960, -4.9853506, -3.6337309, -0.3887089, 0.3892727
5: -3.8390465, -2.4877954, -3.8390503, -2.4877901, -0.4009537, 0.4013082
6: -4.0661411, -2.4190316, -4.0661440, -2.4190323, -0.7641041, 0.7685789
7: -4.2699537, -2.5546417, -4.2699594, -2.5546470, -0.4952250, 0.5064504
8: 0.7713040, 1.2036009, 0.7713028, 1.2036005, -0.1532520, 0.1519969
9: -1.2356480, -0.5044900, -1.2356486, -0.5045778, -0.4410350, 0.4467251

Time for backsubstitution: 6.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3513
type: A, layer: 1, pos: 3402
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 360
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 3390
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3156
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3524
type: A, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3528

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0509873, upper bound: 0.0511025
time: 40.06 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511017, upper bound: 0.0511034
time: 25.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 72.15 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 72.15
Output dim: 8, lower bound: -0.0509879, upper bound: 0.0507255
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 72.15
Output dim: 8, lower bound: -0.0511022, upper bound: 0.0507295
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 72.15
Output dim: 8, lower bound: -0.0507817, upper bound: 0.0510762
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 72.15
Output dim: 8, lower bound: -0.0508939, upper bound: 0.0510791
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 72.15
Output dim: 8, lower bound: -0.0509873, upper bound: 0.0511025
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 72.15
Output dim: 8, lower bound: -0.0511017, upper bound: 0.0511034

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.6666991, 0.1476942, -0.6670349, 0.1477002, -0.7915360, 0.7891380
1: -0.2384364, 0.2703096, -0.2387404, 0.2702796, -0.1665492, 0.1663011
2: -4.1239719, -3.1941748, -4.1239071, -3.1938453, -0.2614151, 0.2571564
3: -3.7640567, -2.3571866, -3.7648215, -2.3565803, -0.5299025, 0.5299181
4: -4.9856963, -3.6337867, -4.9851193, -3.6337323, -0.3872259, 0.3885421
5: -3.8385682, -2.4886646, -3.8387208, -2.4883981, -0.3996777, 0.3992195
6: -4.0584383, -2.4224520, -4.0604420, -2.4200783, -0.7568342, 0.7557353
7: -4.2665229, -2.5554352, -4.2673578, -2.5548391, -0.4911116, 0.5053039
8: 0.7726403, 1.1991364, 0.7714065, 1.2001894, -0.1468085, 0.1523789
9: -1.2302749, -0.5061529, -1.2316399, -0.5046729, -0.4382980, 0.4392013

Time for backsubstitution: 6.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3513
type: B, layer: 1, pos: 3402
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 360
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 455
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 3390
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 3156
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 389
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 1113
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3524
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3566

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510976, upper bound: 0.0505530
time: 35.96 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511004, upper bound: 0.0507278
time: 5.13 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.6572482, 0.1468282, -0.6541698, 0.1441633, -0.7783247, 0.7780923
1: -0.2383743, 0.2699723, -0.2395838, 0.2700589, -0.1653904, 0.1650016
2: -4.1235232, -3.1960969, -4.1233668, -3.1970181, -0.2569115, 0.2562612
3: -3.7658949, -2.3592360, -3.7644887, -2.3605084, -0.5290872, 0.5318649
4: -4.9840131, -3.6382747, -4.9829483, -3.6395080, -0.3842732, 0.3840115
5: -3.8392329, -2.4903030, -3.8380756, -2.4914873, -0.3981006, 0.3993726
6: -4.0666356, -2.4246159, -4.0650969, -2.4256074, -0.7569807, 0.7621514
7: -4.2682223, -2.5658603, -4.2656507, -2.5685863, -0.4936804, 0.4915732
8: 0.7730076, 1.2000771, 0.7737405, 1.1995590, -0.1514898, 0.1458395
9: -1.2331837, -0.5074536, -1.2349012, -0.5071415, -0.4341405, 0.4360545

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3513
type: B, layer: 1, pos: 3402
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 360
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 455
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3390
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 3156
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 389
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 1113
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3524
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3566

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0507761, upper bound: 0.0509055
time: 120.65 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0507792, upper bound: 0.0510810
time: 6.08 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.6578972, 0.1472014, -0.6544966, 0.1444713, -0.7792915, 0.7784547
1: -0.2387240, 0.2704609, -0.2395923, 0.2704563, -0.1661315, 0.1647720
2: -4.1237698, -3.1956263, -4.1234217, -3.1966307, -0.2580363, 0.2558764
3: -3.7658834, -2.3591509, -3.7647924, -2.3604712, -0.5305194, 0.5327542
4: -4.9848828, -3.6380680, -4.9836411, -3.6395082, -0.3847255, 0.3845460
5: -3.8390155, -2.4902062, -3.8381162, -2.4914331, -0.3990171, 0.3994144
6: -4.0660009, -2.4245415, -4.0653219, -2.4255939, -0.7614764, 0.7650938
7: -4.2688971, -2.5658176, -4.2661738, -2.5685515, -0.4947906, 0.4919218
8: 0.7716006, 1.2007633, 0.7725968, 1.1995641, -0.1513232, 0.1477468
9: -1.2345613, -0.5054412, -1.2350180, -0.5054653, -0.4372047, 0.4363382

Time for backsubstitution: 6.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3513
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 3402
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 360
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 455
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3390
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 3156
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 389
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 1113
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3524
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3566

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0508906, upper bound: 0.0509083
time: 155.35 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0508951, upper bound: 0.0510771
time: 765.78 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.6672970, 0.1473403, -0.6676190, 0.1474068, -0.7917458, 0.7897146
1: -0.2392864, 0.2698998, -0.2396291, 0.2698941, -0.1667506, 0.1676633
2: -4.1236734, -3.1929579, -4.1238651, -3.1928582, -0.2626340, 0.2574150
3: -3.7660911, -2.3561265, -3.7657828, -2.3560004, -0.5305487, 0.5356305
4: -4.9844403, -3.6340051, -4.9846272, -3.6337333, -0.3879473, 0.3886317
5: -3.8392651, -2.4881406, -3.8390102, -2.4880457, -0.4000234, 0.4005749
6: -4.0667686, -2.4202096, -4.0659184, -2.4199913, -0.7595800, 0.7654354
7: -4.2692738, -2.5548878, -4.2694373, -2.5548444, -0.4939618, 0.5054905
8: 0.7727228, 1.2029136, 0.7724565, 1.2035936, -0.1517667, 0.1500827
9: -1.2342656, -0.5065494, -1.2355270, -0.5062907, -0.4379345, 0.4445525

Time for backsubstitution: 6.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3513
type: B, layer: 1, pos: 3402
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 360
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 455
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 3390
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 3156
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 389
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 1113
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3524
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3566

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0509818, upper bound: 0.0509300
time: 305.33 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0509875, upper bound: 0.0511002
time: 231.49 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.6679627, 0.1477151, -0.6679658, 0.1477162, -0.7927296, 0.7900972
1: -0.2396369, 0.2703917, -0.2396382, 0.2702936, -0.1674961, 0.1674391
2: -4.1239195, -3.1924858, -4.1239195, -3.1924691, -0.2637643, 0.2570347
3: -3.7660799, -2.3560367, -3.7660856, -2.3559582, -0.5319842, 0.5365282
4: -4.9853144, -3.6337965, -4.9853210, -3.6337318, -0.3884103, 0.3891743
5: -3.8390460, -2.4880407, -3.8390503, -2.4879880, -0.4009435, 0.4006258
6: -4.0661387, -2.4201236, -4.0661426, -2.4199688, -0.7640929, 0.7683910
7: -4.2699509, -2.5548325, -4.2699575, -2.5548005, -0.4950936, 0.5058650
8: 0.7713143, 1.2036008, 0.7713114, 1.2036004, -0.1516025, 0.1519927
9: -1.2356458, -0.5045214, -1.2356470, -0.5046030, -0.4410146, 0.4448564

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3513
type: B, layer: 1, pos: 3402
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 360
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 455
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 3390
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 3156
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 389
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 1113
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3524
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3566

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510974, upper bound: 0.0509109
time: 93.59 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511013, upper bound: 0.0511052
time: 7.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 107.13 seconds
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 107.13
Output dim: 8, lower bound: -0.0510976, upper bound: 0.0505530
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 107.13
Output dim: 8, lower bound: -0.0511004, upper bound: 0.0507278
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 107.13
Output dim: 8, lower bound: -0.0507761, upper bound: 0.0509055
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 107.13
Output dim: 8, lower bound: -0.0507792, upper bound: 0.0510810
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 107.13
Output dim: 8, lower bound: -0.0508906, upper bound: 0.0509083
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 107.13
Output dim: 8, lower bound: -0.0508951, upper bound: 0.0510771
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 107.13
Output dim: 8, lower bound: -0.0509818, upper bound: 0.0509300
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 107.13
Output dim: 8, lower bound: -0.0509875, upper bound: 0.0511002
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 107.13
Output dim: 8, lower bound: -0.0510974, upper bound: 0.0509109
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 107.13
Output dim: 8, lower bound: -0.0511013, upper bound: 0.0511052

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.6639718, 0.1475977, -0.6631223, 0.1468890, -0.7872447, 0.7850634
1: -0.2383576, 0.2697633, -0.2385048, 0.2695092, -0.1657727, 0.1656406
2: -4.1235280, -3.1942730, -4.1233082, -3.1939235, -0.2608890, 0.2564848
3: -3.7636344, -2.3586383, -3.7639627, -2.3586652, -0.5278369, 0.5282025
4: -4.9826050, -3.6338055, -4.9807024, -3.6339436, -0.3838371, 0.3843288
5: -3.8385124, -2.4895561, -3.8385203, -2.4896793, -0.3982813, 0.3981214
6: -4.0581288, -2.4255991, -4.0594397, -2.4245665, -0.7521905, 0.7515815
7: -4.2639256, -2.5554833, -4.2636580, -2.5550413, -0.4883404, 0.5015957
8: 0.7742745, 1.1991069, 0.7737363, 1.1997823, -0.1447910, 0.1500520
9: -1.2300256, -0.5072690, -1.2310871, -0.5062579, -0.4365752, 0.4377115

Time for backsubstitution: 6.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3513
type: A, layer: 1, pos: 3402
type: A, layer: 1, pos: 360
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 3390
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3156
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3524
type: A, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0509784, upper bound: 0.0505539
time: 5.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510933, upper bound: 0.0505503
time: 12.29 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.6666960, 0.1476940, -0.6670297, 0.1477001, -0.7924557, 0.7890438
1: -0.2384363, 0.2703091, -0.2387401, 0.2702788, -0.1660900, 0.1662775
2: -4.1239614, -3.1941750, -4.1238899, -3.1938460, -0.2613744, 0.2569423
3: -3.7640522, -2.3571920, -3.7648149, -2.3565893, -0.5298688, 0.5299743
4: -4.9856405, -3.6337867, -4.9850302, -3.6337323, -0.3872004, 0.3844910
5: -3.8385677, -2.4886680, -3.8387198, -2.4884031, -0.3988311, 0.3992180
6: -4.0584354, -2.4224610, -4.0604367, -2.4200935, -0.7567665, 0.7559785
7: -4.2664704, -2.5554359, -4.2672734, -2.5548396, -0.4910592, 0.5024821
8: 0.7726409, 1.1991364, 0.7714075, 1.2001891, -0.1468066, 0.1500114
9: -1.2302732, -0.5061659, -1.2316370, -0.5046939, -0.4376600, 0.4391865

Time for backsubstitution: 6.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3513
type: A, layer: 1, pos: 3402
type: A, layer: 1, pos: 360
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 3390
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3156
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3524
type: A, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0509835, upper bound: 0.0507235
time: 7.10 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510989, upper bound: 0.0507260
time: 5.52 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.6572447, 0.1468282, -0.6541644, 0.1441631, -0.7792445, 0.7779979
1: -0.2383744, 0.2699718, -0.2395836, 0.2700582, -0.1649418, 0.1649773
2: -4.1235127, -3.1960967, -4.1233501, -3.1970181, -0.2568702, 0.2560543
3: -3.7658908, -2.3592415, -3.7644811, -2.3605180, -0.5290536, 0.5319211
4: -4.9839578, -3.6382749, -4.9828596, -3.6395075, -0.3842477, 0.3799645
5: -3.8392334, -2.4903064, -3.8380747, -2.4914923, -0.3972540, 0.3993711
6: -4.0666313, -2.4246247, -4.0650916, -2.4256232, -0.7569162, 0.7623972
7: -4.2681699, -2.5658607, -4.2655659, -2.5685875, -0.4936279, 0.4887509
8: 0.7730082, 1.2000771, 0.7737415, 1.1995587, -0.1514880, 0.1434720
9: -1.2331817, -0.5074665, -1.2348986, -0.5071623, -0.4335025, 0.4360397

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3513
type: A, layer: 1, pos: 3402
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 360
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3390
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3156
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3524
type: A, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0506630, upper bound: 0.0509031
time: 44.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0507753, upper bound: 0.0510719
time: 21.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.6578938, 0.1472015, -0.6544911, 0.1444712, -0.7802113, 0.7783604
1: -0.2387237, 0.2704605, -0.2395921, 0.2704557, -0.1656832, 0.1647477
2: -4.1237593, -3.1956267, -4.1234040, -3.1966310, -0.2579949, 0.2556698
3: -3.7658787, -2.3591561, -3.7647853, -2.3604798, -0.5304860, 0.5328104
4: -4.9848280, -3.6380680, -4.9835525, -3.6395082, -0.3847000, 0.3804988
5: -3.8390145, -2.4902093, -3.8381155, -2.4914379, -0.3981706, 0.3994130
6: -4.0659981, -2.4245510, -4.0653167, -2.4256091, -0.7614123, 0.7653399
7: -4.2688451, -2.5658178, -4.2660890, -2.5685525, -0.4947382, 0.4890994
8: 0.7716010, 1.2007633, 0.7725976, 1.1995640, -0.1513214, 0.1453793
9: -1.2345595, -0.5054543, -1.2350153, -0.5054860, -0.4365668, 0.4363232

Time for backsubstitution: 6.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3513
type: A, layer: 1, pos: 3402
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 360
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3390
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3156
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3524
type: A, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0507756, upper bound: 0.0510758
time: 51.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0508906, upper bound: 0.0510796
time: 6.30 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.6672937, 0.1473403, -0.6676137, 0.1474068, -0.7926656, 0.7896204
1: -0.2392863, 0.2698993, -0.2396288, 0.2698932, -0.1662934, 0.1676396
2: -4.1236629, -3.1929581, -4.1238484, -3.1928582, -0.2625932, 0.2572016
3: -3.7660873, -2.3561327, -3.7657747, -2.3560092, -0.5305151, 0.5356868
4: -4.9843850, -3.6340051, -4.9845386, -3.6337333, -0.3879218, 0.3845809
5: -3.8392649, -2.4881439, -3.8390093, -2.4880509, -0.3991769, 0.4005734
6: -4.0667663, -2.4202192, -4.0659132, -2.4200068, -0.7595121, 0.7656782
7: -4.2692208, -2.5548882, -4.2693524, -2.5548453, -0.4939094, 0.5026685
8: 0.7727234, 1.2029135, 0.7724574, 1.2035935, -0.1517649, 0.1477152
9: -1.2342640, -0.5065625, -1.2355242, -0.5063117, -0.4372967, 0.4445377

Time for backsubstitution: 6.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3513
type: A, layer: 1, pos: 3402
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 360
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 3390
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3156
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3524
type: A, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0508690, upper bound: 0.0511009
time: 6.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0509826, upper bound: 0.0511019
time: 28.08 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.6652359, 0.1476186, -0.6640537, 0.1469049, -0.7884371, 0.7860209
1: -0.2395577, 0.2698452, -0.2394027, 0.2695236, -0.1667204, 0.1667804
2: -4.1234760, -3.1925836, -4.1233211, -3.1925468, -0.2632390, 0.2563630
3: -3.7656732, -2.3574877, -3.7652340, -2.3580422, -0.5299206, 0.5348192
4: -4.9822235, -3.6338148, -4.9809055, -3.6339426, -0.3850210, 0.3849609
5: -3.8389921, -2.4889328, -3.8388500, -2.4892693, -0.3995475, 0.3995276
6: -4.0658383, -2.4232705, -4.0651474, -2.4244571, -0.7594535, 0.7642401
7: -4.2673545, -2.5548823, -4.2662587, -2.5550053, -0.4923265, 0.5021548
8: 0.7729485, 1.2035713, 0.7736411, 1.2031934, -0.1495854, 0.1496657
9: -1.2354015, -0.5056376, -1.2350979, -0.5061873, -0.4392941, 0.4433681

Time for backsubstitution: 6.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3513
type: A, layer: 1, pos: 3402
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 360
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 3390
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3156
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3524
type: A, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0509783, upper bound: 0.0509302
time: 7.10 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510941, upper bound: 0.0509290
time: 133.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.6679595, 0.1477149, -0.6679606, 0.1477161, -0.7936491, 0.7900028
1: -0.2396367, 0.2703912, -0.2396377, 0.2702929, -0.1670392, 0.1674154
2: -4.1239095, -3.1924860, -4.1239028, -3.1924694, -0.2637236, 0.2568216
3: -3.7660758, -2.3560429, -3.7660785, -2.3559673, -0.5319506, 0.5365845
4: -4.9852600, -3.6337965, -4.9852333, -3.6337318, -0.3883848, 0.3851233
5: -3.8390462, -2.4880438, -3.8390496, -2.4879930, -0.4000971, 0.4006243
6: -4.0661354, -2.4201326, -4.0661373, -2.4199841, -0.7640254, 0.7686342
7: -4.2698989, -2.5548332, -4.2698722, -2.5548010, -0.4950412, 0.5030433
8: 0.7713149, 1.2036006, 0.7713122, 1.2036002, -0.1516006, 0.1496251
9: -1.2356440, -0.5045344, -1.2356441, -0.5046239, -0.4403767, 0.4448416

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3513
type: A, layer: 1, pos: 3402
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 360
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 3390
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3175
type: A, layer: 1, pos: 3156
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3524
type: A, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0509827, upper bound: 0.0510968
time: 396.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510984, upper bound: 0.0509025
time: 150.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 553.88 seconds
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0509784, upper bound: 0.0505539
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0510933, upper bound: 0.0505503
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0509835, upper bound: 0.0507235
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0510989, upper bound: 0.0507260
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0506630, upper bound: 0.0509031
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0507753, upper bound: 0.0510719
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0507756, upper bound: 0.0510758
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0508906, upper bound: 0.0510796
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0508690, upper bound: 0.0511009
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0509826, upper bound: 0.0511019
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0509783, upper bound: 0.0509302
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0510941, upper bound: 0.0509290
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0509827, upper bound: 0.0510968
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 553.88
Output dim: 8, lower bound: -0.0510984, upper bound: 0.0509025

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.6661248, 0.1477472, -0.6630771, 0.1468613, -0.7896127, 0.7848742
1: -0.2410913, 0.2697316, -0.2384761, 0.2694806, -0.1678139, 0.1649660
2: -4.1235309, -3.1914616, -4.1232944, -3.1939266, -0.2613808, 0.2574483
3: -3.7636497, -2.3582640, -3.7639580, -2.3586938, -0.5276194, 0.5284028
4: -4.9828396, -3.6317871, -4.9806466, -3.6339436, -0.3838339, 0.3848042
5: -3.8390422, -2.4915817, -3.8385208, -2.4914441, -0.3992911, 0.3970461
6: -4.0580096, -2.4141073, -4.0592742, -2.4245670, -0.7499429, 0.7598165
7: -4.2698731, -2.5576065, -4.2636414, -2.5569165, -0.4962667, 0.4987166
8: 0.7741506, 1.1991992, 0.7738057, 1.1997709, -0.1447312, 0.1502111
9: -1.2304646, -0.5072657, -1.2310835, -0.5063041, -0.4356704, 0.4389075

Time for backsubstitution: 6.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3513
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 3402
type: B, layer: 1, pos: 360
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 455
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 3390
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 3156
type: B, layer: 1, pos: 3175
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 389
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 1113
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3524
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 399

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0508186, upper bound: 0.0505517
time: 91.48 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0508188, upper bound: 0.0505513
time: 37.28 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 30.38 + 3608.49 = 3638.87 seconds
