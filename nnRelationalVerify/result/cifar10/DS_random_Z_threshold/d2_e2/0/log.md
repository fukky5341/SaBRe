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
execution time: IAR + RelationalAnalysis = 9.77 + 23.22 = 32.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0511057, upper bound: 0.0511097

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3390
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 3131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3390

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511051, upper bound: 0.0511059
time: 83.09 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511062, upper bound: 0.0511061
time: 86.88 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 169.99 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 169.99
Output dim: 8, lower bound: -0.0511051, upper bound: 0.0511059
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 169.99
Output dim: 8, lower bound: -0.0511062, upper bound: 0.0511061

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930704, 0.7930717
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1681813, 0.1681808
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2614945, 0.2614953
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5348053, 0.5348046
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3903351, 0.3903354
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4016389, 0.4016378
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7665716, 0.7665707
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5072622, 0.5072624
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1570034, 0.1570028
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442397, 0.4442389

Time for backsubstitution: 6.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 869

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511031, upper bound: 0.0511096
time: 5.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511031, upper bound: 0.0511077
time: 95.17 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930717, 0.7930704
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1681808, 0.1681813
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2614953, 0.2614944
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5348046, 0.5348054
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3903354, 0.3903351
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4016379, 0.4016389
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7665707, 0.7665716
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5072623, 0.5072623
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1570028, 0.1570034
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442389, 0.4442396

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3273

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 711

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511042, upper bound: 0.0511066
time: 6.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511052, upper bound: 0.0511103
time: 6.27 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 19.63 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 19.63
Output dim: 8, lower bound: -0.0511031, upper bound: 0.0511096
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 19.63
Output dim: 8, lower bound: -0.0511031, upper bound: 0.0511077
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 19.63
Output dim: 8, lower bound: -0.0511042, upper bound: 0.0511066
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 19.63
Output dim: 8, lower bound: -0.0511052, upper bound: 0.0511103

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930704, 0.7930717
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1681813, 0.1681808
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2614945, 0.2614953
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5348053, 0.5348046
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3903351, 0.3903354
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4016389, 0.4016378
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7665716, 0.7665707
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5072622, 0.5072624
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1570034, 0.1570028
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442397, 0.4442389

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 202

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2544

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510898, upper bound: 0.0511012
time: 107.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511010, upper bound: 0.0510959
time: 7.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930704, 0.7930717
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1681813, 0.1681808
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2614945, 0.2614953
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5348053, 0.5348046
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3903351, 0.3903354
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4016389, 0.4016378
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7665716, 0.7665707
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5072622, 0.5072624
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1570034, 0.1570028
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442397, 0.4442389

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 779

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3513

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510417, upper bound: 0.0511094
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511052, upper bound: 0.0510424
time: 158.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7931041, 0.7931001
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1680425, 0.1680487
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2614448, 0.2614442
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5343850, 0.5344056
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3903089, 0.3903139
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4012204, 0.4012400
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7659953, 0.7660112
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5071182, 0.5071247
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569962, 0.1569968
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442813, 0.4442827

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 794

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1113

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511062, upper bound: 0.0511063
time: 89.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511062, upper bound: 0.0511045
time: 25.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7931013, 0.7931027
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1680482, 0.1680431
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2614450, 0.2614439
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5344049, 0.5343857
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3903142, 0.3903086
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4012389, 0.4012215
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7660103, 0.7659962
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5071248, 0.5071181
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569962, 0.1569969
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442818, 0.4442821

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 737

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2324

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511031, upper bound: 0.0511043
time: 173.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511031, upper bound: 0.0511101
time: 6.45 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 186.19 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 186.19
Output dim: 8, lower bound: -0.0510898, upper bound: 0.0511012
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 186.19
Output dim: 8, lower bound: -0.0511010, upper bound: 0.0510959
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 186.19
Output dim: 8, lower bound: -0.0510417, upper bound: 0.0511094
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 186.19
Output dim: 8, lower bound: -0.0511052, upper bound: 0.0510424
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 186.19
Output dim: 8, lower bound: -0.0511062, upper bound: 0.0511063
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 186.19
Output dim: 8, lower bound: -0.0511062, upper bound: 0.0511045
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 186.19
Output dim: 8, lower bound: -0.0511031, upper bound: 0.0511043
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 186.19
Output dim: 8, lower bound: -0.0511031, upper bound: 0.0511101

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930290, 0.7930316
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1679711, 0.1679876
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2613376, 0.2613164
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5339483, 0.5338763
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3901579, 0.3901195
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4007650, 0.4006841
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7655584, 0.7654871
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5065385, 0.5064844
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569986, 0.1569982
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4440249, 0.4440298

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3025

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2521

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510757, upper bound: 0.0510958
time: 12.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510835, upper bound: 0.0510854
time: 122.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930304, 0.7930301
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1679881, 0.1679706
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2613155, 0.2613384
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5338770, 0.5339476
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3901192, 0.3901582
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4006852, 0.4007639
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7654880, 0.7655574
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5064843, 0.5065386
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569988, 0.1569979
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4440306, 0.4440241

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3276

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3494

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511008, upper bound: 0.0510929
time: 77.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511008, upper bound: 0.0510932
time: 112.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930744, 0.7930751
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1681821, 0.1681815
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2591098, 0.2590140
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5345231, 0.5344999
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3889173, 0.3888705
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4013029, 0.4012812
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7664645, 0.7664593
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5057668, 0.5057190
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1561778, 0.1562171
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4431400, 0.4430933

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 558

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510411, upper bound: 0.0510689
time: 60.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510061, upper bound: 0.0511072
time: 112.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930739, 0.7930757
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1681820, 0.1681816
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2590131, 0.2591106
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5345007, 0.5345223
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3888702, 0.3889176
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4012823, 0.4013019
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7664603, 0.7664636
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5057189, 0.5057669
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1562178, 0.1561771
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4430941, 0.4431392

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2977

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 896

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511029, upper bound: 0.0510450
time: 12.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511029, upper bound: 0.0510447
time: 14.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7931041, 0.7931001
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1680425, 0.1680487
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2614448, 0.2614442
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5343850, 0.5344056
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3903089, 0.3903139
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4012204, 0.4012400
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7659953, 0.7660112
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5071182, 0.5071247
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569962, 0.1569968
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442813, 0.4442827

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 782

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511045, upper bound: 0.0511050
time: 139.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511053, upper bound: 0.0511014
time: 214.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7931041, 0.7931001
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1680425, 0.1680487
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2614448, 0.2614442
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5343850, 0.5344056
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3903089, 0.3903139
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4012204, 0.4012400
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7659953, 0.7660112
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5071182, 0.5071247
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569962, 0.1569968
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442813, 0.4442827

Time for backsubstitution: 5.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2581

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510983, upper bound: 0.0511007
time: 45.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510985, upper bound: 0.0510976
time: 72.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7931013, 0.7931027
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1680482, 0.1680431
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2614450, 0.2614439
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5344049, 0.5343857
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3903142, 0.3903086
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4012389, 0.4012215
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7660103, 0.7659962
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5071248, 0.5071181
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569962, 0.1569969
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442818, 0.4442821

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3100

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2162

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511024, upper bound: 0.0510956
time: 196.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510941, upper bound: 0.0511044
time: 105.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7931013, 0.7931027
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1680482, 0.1680431
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2614450, 0.2614439
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5344049, 0.5343857
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3903142, 0.3903086
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4012389, 0.4012215
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7660103, 0.7659962
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5071248, 0.5071181
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569962, 0.1569969
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442818, 0.4442821

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2663

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510915, upper bound: 0.0510941
time: 158.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510926, upper bound: 0.0510965
time: 54.70 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 219.71 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0510757, upper bound: 0.0510958
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0510835, upper bound: 0.0510854
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0511008, upper bound: 0.0510929
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0511008, upper bound: 0.0510932
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0510411, upper bound: 0.0510689
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0510061, upper bound: 0.0511072
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0511029, upper bound: 0.0510450
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0511029, upper bound: 0.0510447
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0511045, upper bound: 0.0511050
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0511053, upper bound: 0.0511014
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0510983, upper bound: 0.0511007
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0510985, upper bound: 0.0510976
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0511024, upper bound: 0.0510956
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0510941, upper bound: 0.0511044
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0510915, upper bound: 0.0510941
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 219.71
Output dim: 8, lower bound: -0.0510926, upper bound: 0.0510965

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930249, 0.7930283
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1672451, 0.1672357
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2612650, 0.2612306
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5338799, 0.5337992
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3899292, 0.3898776
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4007026, 0.4006138
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7653913, 0.7653039
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5065213, 0.5064685
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569687, 0.1569734
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4436470, 0.4436457

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 541

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2565

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510701, upper bound: 0.0510930
time: 5.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510717, upper bound: 0.0510891
time: 7.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930256, 0.7930276
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1672192, 0.1672616
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2612518, 0.2612437
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5338712, 0.5338079
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3899160, 0.3898908
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4006947, 0.4006218
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7653753, 0.7653201
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5065225, 0.5064673
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569739, 0.1569683
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4436407, 0.4436521

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3100

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510684, upper bound: 0.0510750
time: 30.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510720, upper bound: 0.0510742
time: 6.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930304, 0.7930301
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1679881, 0.1679706
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2613155, 0.2613384
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5338770, 0.5339476
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3901192, 0.3901582
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4006852, 0.4007639
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7654880, 0.7655574
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5064843, 0.5065386
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569988, 0.1569979
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4440306, 0.4440241

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2521

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3511

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510993, upper bound: 0.0510890
time: 6.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510952, upper bound: 0.0510934
time: 6.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930304, 0.7930301
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1679881, 0.1679706
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2613155, 0.2613384
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5338770, 0.5339476
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3901192, 0.3901582
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4006852, 0.4007639
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7654880, 0.7655574
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5064843, 0.5065386
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569988, 0.1569979
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4440306, 0.4440241

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 831

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 898

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511011, upper bound: 0.0510966
time: 6.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511011, upper bound: 0.0510958
time: 6.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930346, 0.7930332
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1681060, 0.1681083
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2569618, 0.2567474
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5352031, 0.5351036
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3876233, 0.3875082
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4010663, 0.4010273
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7650199, 0.7649190
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5050271, 0.5049425
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1558483, 0.1558620
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4432672, 0.4432002

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 899

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1117

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510410, upper bound: 0.0510728
time: 13.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510410, upper bound: 0.0510716
time: 44.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930325, 0.7930353
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1681089, 0.1681054
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2568432, 0.2568659
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5351267, 0.5351800
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3875550, 0.3875766
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4010490, 0.4010446
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7649242, 0.7650148
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5049903, 0.5049793
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1558226, 0.1558876
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4432469, 0.4432205

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 125

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0509885, upper bound: 0.0511032
time: 70.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510032, upper bound: 0.0510894
time: 9.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930739, 0.7930757
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1681820, 0.1681816
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2590131, 0.2591106
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5345007, 0.5345223
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3888702, 0.3889176
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4012823, 0.4013019
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7664603, 0.7664636
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5057189, 0.5057669
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1562178, 0.1561771
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4430941, 0.4431392

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2449

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 778

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511047, upper bound: 0.0510444
time: 6.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511046, upper bound: 0.0510417
time: 76.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930739, 0.7930757
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1681820, 0.1681816
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2590131, 0.2591106
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5345007, 0.5345223
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3888702, 0.3889176
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4012823, 0.4013019
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7664603, 0.7664636
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5057189, 0.5057669
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1562178, 0.1561771
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4430941, 0.4431392

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2162

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2521

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510878, upper bound: 0.0510350
time: 84.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510965, upper bound: 0.0510253
time: 91.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930887, 0.7930846
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1680002, 0.1680054
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2613191, 0.2613308
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5337521, 0.5338899
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3902008, 0.3902136
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4004266, 0.4005638
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7654029, 0.7654791
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5061591, 0.5062624
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569882, 0.1569888
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442061, 0.4442192

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2162

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511032, upper bound: 0.0510992
time: 6.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510949, upper bound: 0.0511076
time: 6.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930886, 0.7930847
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1679992, 0.1680064
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2613315, 0.2613185
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5338693, 0.5337727
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3902086, 0.3902058
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4005443, 0.4004461
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7654632, 0.7654188
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5062559, 0.5061656
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569882, 0.1569888
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442178, 0.4442075

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3004

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 141

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510938, upper bound: 0.0510915
time: 21.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510895, upper bound: 0.0510944
time: 86.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930998, 0.7930959
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1678620, 0.1678369
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2613707, 0.2613813
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5343570, 0.5343817
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3902090, 0.3902280
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4011806, 0.4012046
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7657480, 0.7657999
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5071013, 0.5071111
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569620, 0.1569632
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4441038, 0.4440829

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2656

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510899, upper bound: 0.0510923
time: 11.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510902, upper bound: 0.0510876
time: 48.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930999, 0.7930958
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1678307, 0.1678682
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2613819, 0.2613701
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5343611, 0.5343776
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3902230, 0.3902140
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4011851, 0.4012001
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7657841, 0.7657639
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5071046, 0.5071078
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569626, 0.1569626
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4440815, 0.4441053

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2095

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2543

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510957, upper bound: 0.0510911
time: 106.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510941, upper bound: 0.0510985
time: 7.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930857, 0.7930905
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1676275, 0.1676339
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2604691, 0.2604135
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5284467, 0.5282754
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3890575, 0.3889970
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3940126, 0.3938247
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7615912, 0.7615336
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5014044, 0.5013314
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1568589, 0.1568555
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4438529, 0.4438753

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3045

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2130

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510940, upper bound: 0.0510908
time: 25.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510967, upper bound: 0.0510886
time: 64.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930891, 0.7930872
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1676390, 0.1676224
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2604145, 0.2604681
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5282946, 0.5284275
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3890026, 0.3890519
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.3938421, 0.3939952
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7615476, 0.7615772
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5013382, 0.5013977
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1568548, 0.1568596
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4438750, 0.4438531

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3513

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3052

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510512, upper bound: 0.0511046
time: 31.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510924, upper bound: 0.0510533
time: 120.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.6679966, 0.1477210, -0.6679966, 0.1477210, -0.7930930, 0.7930945
1: -0.2396422, 0.2706149, -0.2396422, 0.2706149, -0.1680283, 0.1680223
2: -4.1239200, -3.1923556, -4.1239200, -3.1923556, -0.2611948, 0.2611964
3: -3.7661233, -2.3556261, -3.7661233, -2.3556261, -0.5340976, 0.5341096
4: -4.9853544, -3.6335359, -4.9853544, -3.6335359, -0.3895274, 0.3895484
5: -3.8390603, -2.4877713, -3.8390603, -2.4877713, -0.4009393, 0.4009569
6: -4.0661535, -2.4190209, -4.0661535, -2.4190209, -0.7659019, 0.7658964
7: -4.2699785, -2.5546284, -4.2699785, -2.5546284, -0.5069071, 0.5069039
8: 0.7712983, 1.2036116, 0.7712983, 1.2036116, -0.1569672, 0.1569688
9: -1.2356516, -0.5042349, -1.2356516, -0.5042349, -0.4442811, 0.4442813

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 360
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3156
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 1097
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 399
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 1095
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3402
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2556

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3273

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510899, upper bound: 0.0510987
time: 6.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510903, upper bound: 0.0510927
time: 102.70 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 115.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510701, upper bound: 0.0510930
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510717, upper bound: 0.0510891
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510684, upper bound: 0.0510750
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510720, upper bound: 0.0510742
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510993, upper bound: 0.0510890
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510952, upper bound: 0.0510934
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0511011, upper bound: 0.0510966
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0511011, upper bound: 0.0510958
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510410, upper bound: 0.0510728
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510410, upper bound: 0.0510716
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0509885, upper bound: 0.0511032
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510032, upper bound: 0.0510894
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0511047, upper bound: 0.0510444
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0511046, upper bound: 0.0510417
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510878, upper bound: 0.0510350
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510965, upper bound: 0.0510253
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0511032, upper bound: 0.0510992
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510949, upper bound: 0.0511076
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510938, upper bound: 0.0510915
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510895, upper bound: 0.0510944
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510899, upper bound: 0.0510923
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510902, upper bound: 0.0510876
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510957, upper bound: 0.0510911
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510941, upper bound: 0.0510985
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510940, upper bound: 0.0510908
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510967, upper bound: 0.0510886
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510512, upper bound: 0.0511046
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510924, upper bound: 0.0510533
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510899, upper bound: 0.0510987
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 115.99
Output dim: 8, lower bound: -0.0510903, upper bound: 0.0510927
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 115.99
Output dim: 8, lower bound: -0.0510926, upper bound: 0.0510965

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 32.99 + 3676.13 = 3709.12 seconds
