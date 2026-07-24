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
execution time: IAR + RelationalAnalysis = 11.69 + 30.11 = 41.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0511057, upper bound: 0.0511097

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 390

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511034, upper bound: 0.0509277
time: 11.66 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0509232, upper bound: 0.0511086
time: 6.32 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 18.06 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 18.06
Output dim: 8, lower bound: -0.0511034, upper bound: 0.0509277
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 18.06
Output dim: 8, lower bound: -0.0509232, upper bound: 0.0511086

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7904717, 0.7905993
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1679377, 0.1679505
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2638144, 0.2635089
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5324582, 0.5322332
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3887185, 0.3883206
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4009585, 0.4007338
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7646278, 0.7643701
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4937614, 0.4929992
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1532618, 0.1530433
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4465733, 0.4467366

Time for backsubstitution: 7.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 399

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511018, upper bound: 0.0505442
time: 99.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0507251, upper bound: 0.0509241
time: 10.41 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7905993, 0.7904717
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1679505, 0.1679377
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2635089, 0.2638144
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5322333, 0.5324582
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3883206, 0.3887185
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4007338, 0.4009585
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7643700, 0.7646278
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4929992, 0.4937614
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1530433, 0.1532618
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4467366, 0.4465733

Time for backsubstitution: 6.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 399

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0509214, upper bound: 0.0507248
time: 157.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0505437, upper bound: 0.0511029
time: 97.59 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 261.43 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 261.43
Output dim: 8, lower bound: -0.0511018, upper bound: 0.0505442
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 261.43
Output dim: 8, lower bound: -0.0507251, upper bound: 0.0509241
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 261.43
Output dim: 8, lower bound: -0.0509214, upper bound: 0.0507248
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 261.43
Output dim: 8, lower bound: -0.0505437, upper bound: 0.0511029

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7902418, 0.7904086
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1675133, 0.1675422
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2612821, 0.2609059
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5344521, 0.5342540
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3881186, 0.3877418
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4009594, 0.4007374
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7668651, 0.7665151
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4949822, 0.4945495
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1482616, 0.1477779
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4437734, 0.4440773

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3323

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510973, upper bound: 0.0505414
time: 5.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510982, upper bound: 0.0505428
time: 42.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7904085, 0.7902418
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1675422, 0.1675133
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2609060, 0.2612821
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5342539, 0.5344521
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3877418, 0.3881186
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4007374, 0.4009594
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7665151, 0.7668651
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4945495, 0.4949821
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1477779, 0.1482616
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4440772, 0.4437734

Time for backsubstitution: 5.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3323

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0505403, upper bound: 0.0511011
time: 33.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0505383, upper bound: 0.0511028
time: 150.86 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 189.91 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 189.91
Output dim: 8, lower bound: -0.0510973, upper bound: 0.0505414
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 189.91
Output dim: 8, lower bound: -0.0510982, upper bound: 0.0505428
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 189.91
Output dim: 8, lower bound: -0.0505403, upper bound: 0.0511011
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 189.91
Output dim: 8, lower bound: -0.0505383, upper bound: 0.0511028

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7902122, 0.7903798
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1675117, 0.1675406
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2607989, 0.2604169
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5341537, 0.5339407
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3874130, 0.3870459
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4002932, 0.4000606
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7667459, 0.7663860
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4949163, 0.4944908
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1482587, 0.1477739
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4437156, 0.4440218

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3093

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510887, upper bound: 0.0505299
time: 94.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510920, upper bound: 0.0505330
time: 7.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7902130, 0.7903789
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1675117, 0.1675406
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2607931, 0.2604228
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5341389, 0.5339554
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3874227, 0.3870363
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4002826, 0.4000712
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7667360, 0.7663959
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4949235, 0.4944836
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1482576, 0.1477751
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4437181, 0.4440194

Time for backsubstitution: 5.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3093

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510888, upper bound: 0.0505372
time: 6.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510895, upper bound: 0.0505322
time: 57.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7903789, 0.7902130
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1675406, 0.1675117
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2604228, 0.2607931
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5339554, 0.5341389
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3870363, 0.3874227
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4000711, 0.4002827
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7663959, 0.7667359
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4944835, 0.4949234
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1477751, 0.1482576
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4440194, 0.4437181

Time for backsubstitution: 6.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3093

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0505288, upper bound: 0.0510898
time: 226.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0505340, upper bound: 0.0510901
time: 133.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7903798, 0.7902122
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1675406, 0.1675117
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2604169, 0.2607989
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5339407, 0.5341536
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3870459, 0.3874130
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4000606, 0.4002932
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7663860, 0.7667459
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4944908, 0.4949163
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1477739, 0.1482587
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4440218, 0.4437156

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3093

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0505298, upper bound: 0.0510915
time: 74.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0505303, upper bound: 0.0510893
time: 150.14 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 231.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 231.49
Output dim: 8, lower bound: -0.0510887, upper bound: 0.0505299
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 231.49
Output dim: 8, lower bound: -0.0510920, upper bound: 0.0505330
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 231.49
Output dim: 8, lower bound: -0.0510888, upper bound: 0.0505372
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 231.49
Output dim: 8, lower bound: -0.0510895, upper bound: 0.0505322
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 231.49
Output dim: 8, lower bound: -0.0505288, upper bound: 0.0510898
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 231.49
Output dim: 8, lower bound: -0.0505340, upper bound: 0.0510901
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 231.49
Output dim: 8, lower bound: -0.0505298, upper bound: 0.0510915
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 231.49
Output dim: 8, lower bound: -0.0505303, upper bound: 0.0510893

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7901638, 0.7903298
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1670528, 0.1670705
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2593578, 0.2590747
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5314933, 0.5315827
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3858157, 0.3855197
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3969495, 0.3969954
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7651459, 0.7650115
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4920302, 0.4918799
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1481750, 0.1476978
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4432603, 0.4435874

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3528

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0509751, upper bound: 0.0505349
time: 7.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510890, upper bound: 0.0504194
time: 101.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7901622, 0.7903313
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1670416, 0.1670817
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2594568, 0.2589758
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5317957, 0.5312803
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3858869, 0.3854486
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3972279, 0.3967170
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7653714, 0.7647859
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4923054, 0.4916047
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1481826, 0.1476901
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4432812, 0.4435665

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3528

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0509774, upper bound: 0.0505276
time: 106.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510913, upper bound: 0.0504147
time: 204.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7901646, 0.7903289
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1670528, 0.1670705
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2593520, 0.2590807
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5314785, 0.5315974
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3858254, 0.3855101
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3969390, 0.3970059
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7651359, 0.7650214
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4920375, 0.4918727
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1481738, 0.1476989
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4432628, 0.4435849

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3528

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0509726, upper bound: 0.0505360
time: 7.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510880, upper bound: 0.0504218
time: 8.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7901630, 0.7903305
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1670417, 0.1670817
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2594510, 0.2589817
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5317810, 0.5312949
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3858965, 0.3854390
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3972174, 0.3967275
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7653615, 0.7647958
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4923126, 0.4915975
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1481815, 0.1476913
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4432836, 0.4435641

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3528

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0509753, upper bound: 0.0505313
time: 21.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510887, upper bound: 0.0504171
time: 46.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7903305, 0.7901630
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1670817, 0.1670417
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2589817, 0.2594509
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5312949, 0.5317810
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3854390, 0.3858965
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3967275, 0.3972174
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7647958, 0.7653615
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4915975, 0.4923126
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1476913, 0.1481814
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4435641, 0.4432836

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3528

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504162, upper bound: 0.0510874
time: 117.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0505294, upper bound: 0.0509750
time: 37.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7903289, 0.7901646
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1670705, 0.1670528
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2590807, 0.2593520
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5315975, 0.5314785
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3855101, 0.3858254
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3970059, 0.3969390
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7650214, 0.7651359
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4918727, 0.4920375
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1476990, 0.1481738
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4435850, 0.4432627

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3528

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504179, upper bound: 0.0510922
time: 6.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0505319, upper bound: 0.0505956
time: 120.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7903313, 0.7901622
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1670817, 0.1670416
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2589758, 0.2594568
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5312803, 0.5317957
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3854486, 0.3858869
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3967170, 0.3972279
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7647859, 0.7653714
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4916048, 0.4923053
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1476901, 0.1481826
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4435665, 0.4432812

Time for backsubstitution: 6.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3528

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504137, upper bound: 0.0510910
time: 21.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0505279, upper bound: 0.0509770
time: 75.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7903298, 0.7901638
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1670705, 0.1670528
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2590747, 0.2593578
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5315828, 0.5314932
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3855197, 0.3858157
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3969954, 0.3969495
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7650115, 0.7651458
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4918799, 0.4920302
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1476978, 0.1481750
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4435874, 0.4432603

Time for backsubstitution: 6.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3528

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504159, upper bound: 0.0510882
time: 20.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0505293, upper bound: 0.0509799
time: 7.44 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 34.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0509751, upper bound: 0.0505349
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0510890, upper bound: 0.0504194
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0509774, upper bound: 0.0505276
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0510913, upper bound: 0.0504147
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0509726, upper bound: 0.0505360
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0510880, upper bound: 0.0504218
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0509753, upper bound: 0.0505313
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0510887, upper bound: 0.0504171
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0504162, upper bound: 0.0510874
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0505294, upper bound: 0.0509750
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0504179, upper bound: 0.0510922
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0505319, upper bound: 0.0505956
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0504137, upper bound: 0.0510910
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0505279, upper bound: 0.0509770
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0504159, upper bound: 0.0510882
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 34.65
Output dim: 8, lower bound: -0.0505293, upper bound: 0.0509799

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7898238, 0.7900089
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1663003, 0.1663558
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2581935, 0.2579296
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5312763, 0.5313607
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3855625, 0.3852431
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3962904, 0.3963213
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7649746, 0.7648373
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4915408, 0.4913844
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1465306, 0.1459688
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4413395, 0.4417396

Time for backsubstitution: 6.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2448

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510854, upper bound: 0.0504099
time: 6.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510774, upper bound: 0.0504136
time: 28.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7898222, 0.7900103
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1662891, 0.1663669
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2582925, 0.2578293
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5315789, 0.5310581
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3856295, 0.3851719
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3965688, 0.3960426
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7652001, 0.7646116
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4918160, 0.4911090
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1465380, 0.1459611
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4413604, 0.4417180

Time for backsubstitution: 6.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2448

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510871, upper bound: 0.0504045
time: 48.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510780, upper bound: 0.0504128
time: 7.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7898247, 0.7900081
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1663003, 0.1663558
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2581877, 0.2579355
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5312616, 0.5313754
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3855722, 0.3852335
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3962798, 0.3963318
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7649646, 0.7648472
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4915481, 0.4913771
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1465294, 0.1459699
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4413419, 0.4417371

Time for backsubstitution: 6.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2448

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510837, upper bound: 0.0504090
time: 121.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510756, upper bound: 0.0504184
time: 12.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7898231, 0.7900095
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1662892, 0.1663669
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2582867, 0.2578353
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5315641, 0.5310727
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3856391, 0.3851623
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3965583, 0.3960531
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7651902, 0.7646215
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4918232, 0.4911018
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1465368, 0.1459623
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4413628, 0.4417156

Time for backsubstitution: 6.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2448

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510855, upper bound: 0.0504070
time: 8.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510765, upper bound: 0.0504123
time: 124.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7900094, 0.7898231
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1663669, 0.1662891
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2578353, 0.2582866
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5310727, 0.5315641
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3851623, 0.3856391
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3960531, 0.3965583
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7646216, 0.7651902
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4911018, 0.4918232
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1459623, 0.1465369
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4417156, 0.4413628

Time for backsubstitution: 6.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2448

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504120, upper bound: 0.0510776
time: 103.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504031, upper bound: 0.0510843
time: 34.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7900081, 0.7898246
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1663558, 0.1663003
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2579355, 0.2581877
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5313753, 0.5312617
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3852335, 0.3855721
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3963318, 0.3962799
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7648472, 0.7649647
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4913772, 0.4915481
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1459699, 0.1465294
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4417372, 0.4413419

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2448

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504148, upper bound: 0.0510791
time: 63.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504052, upper bound: 0.0510865
time: 40.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7900103, 0.7898222
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1663669, 0.1662891
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2578294, 0.2582925
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5310581, 0.5315789
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3851719, 0.3856294
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3960426, 0.3965688
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7646117, 0.7652001
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4911090, 0.4918160
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1459611, 0.1465380
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4417180, 0.4413604

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2448

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504092, upper bound: 0.0510836
time: 13.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504015, upper bound: 0.0510920
time: 7.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7900089, 0.7898238
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1663558, 0.1663003
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2579296, 0.2581935
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5313607, 0.5312765
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3852431, 0.3855625
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3963213, 0.3962904
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7648373, 0.7649746
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4913845, 0.4915408
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1459688, 0.1465306
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4417396, 0.4413395

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2448

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504130, upper bound: 0.0510770
time: 21.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504036, upper bound: 0.0510904
time: 6.93 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 34.77 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0510854, upper bound: 0.0504099
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0510774, upper bound: 0.0504136
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0510871, upper bound: 0.0504045
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0510780, upper bound: 0.0504128
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0510837, upper bound: 0.0504090
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0510756, upper bound: 0.0504184
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0510855, upper bound: 0.0504070
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0510765, upper bound: 0.0504123
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0504120, upper bound: 0.0510776
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0504031, upper bound: 0.0510843
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0504148, upper bound: 0.0510791
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0504052, upper bound: 0.0510865
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0504092, upper bound: 0.0510836
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0504015, upper bound: 0.0510920
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0504130, upper bound: 0.0510770
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 34.77
Output dim: 8, lower bound: -0.0504036, upper bound: 0.0510904

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7898048, 0.7899897
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1658634, 0.1659402
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2572796, 0.2570528
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5303007, 0.5304328
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3842153, 0.3839497
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3954427, 0.3955179
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7644012, 0.7642941
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4905615, 0.4904493
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1464245, 0.1458540
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4408247, 0.4412431

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3311

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510847, upper bound: 0.0504019
time: 47.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510806, upper bound: 0.0504042
time: 144.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7898046, 0.7899899
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1658847, 0.1659189
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2573167, 0.2570156
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5303485, 0.5303851
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3842692, 0.3838958
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3954870, 0.3954737
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7644314, 0.7642639
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4906057, 0.4904051
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1464158, 0.1458627
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4408431, 0.4412248

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3311

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510768, upper bound: 0.0504103
time: 93.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510719, upper bound: 0.0504142
time: 106.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7898033, 0.7899911
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1658522, 0.1659513
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2573785, 0.2569525
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5306032, 0.5301302
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3842822, 0.3838786
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3957211, 0.3952392
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7646267, 0.7640685
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.4908367, 0.4901738
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1464319, 0.1458463
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4408457, 0.4412216

Time for backsubstitution: 6.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3311

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510866, upper bound: 0.0503987
time: 578.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510820, upper bound: 0.0504030
time: 6.15 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 591.85 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 591.85
Output dim: 8, lower bound: -0.0510847, upper bound: 0.0504019
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 591.85
Output dim: 8, lower bound: -0.0510806, upper bound: 0.0504042
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 591.85
Output dim: 8, lower bound: -0.0510768, upper bound: 0.0504103
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 591.85
Output dim: 8, lower bound: -0.0510719, upper bound: 0.0504142
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 591.85
Output dim: 8, lower bound: -0.0510866, upper bound: 0.0503987
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 591.85
Output dim: 8, lower bound: -0.0510820, upper bound: 0.0504030
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0510780, upper bound: 0.0504128
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0510837, upper bound: 0.0504090
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0510756, upper bound: 0.0504184
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0510855, upper bound: 0.0504070
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0510765, upper bound: 0.0504123
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0504120, upper bound: 0.0510776
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0504031, upper bound: 0.0510843
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0504148, upper bound: 0.0510791
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0504052, upper bound: 0.0510865
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0504092, upper bound: 0.0510836
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0504015, upper bound: 0.0510920
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0504130, upper bound: 0.0510770
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 591.85
Output dim: 8, lower bound: -0.0504036, upper bound: 0.0510904

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 41.80 + 4074.29 = 4116.09 seconds
