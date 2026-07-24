## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0396018585


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1075137, 0.1075136)
1: (0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0940093, 0.0940093)
2: (-4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1767667, 0.1767667)
3: (-6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2552857, 0.2552858)
4: (-6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2806582, 0.2806582)
5: (-6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3037086, 0.3037086)
6: (-8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3028632, 0.3028633)
7: (-4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8710415, 0.8710414)
8: (-0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2802931, 0.2802931)
9: (0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0738531, 0.0738531)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.74 + 20.05 = 27.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0396317, upper bound: 0.0396418

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3512

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3108

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396135, upper bound: 0.0395590
time: 363.57 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395546, upper bound: 0.0396169
time: 29.40 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 393.04 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 393.04
Output dim: 1, lower bound: -0.0396135, upper bound: 0.0395590
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 393.04
Output dim: 1, lower bound: -0.0395546, upper bound: 0.0396169

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1074012, 0.1074390
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0939399, 0.0938909
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1764674, 0.1765892
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2552119, 0.2552188
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2803135, 0.2804402
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3036054, 0.3036165
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3028464, 0.3028462
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8707548, 0.8708397
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2802649, 0.2802457
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0738287, 0.0738200

Time for backsubstitution: 5.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3512

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2657

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395936, upper bound: 0.0395471
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395930, upper bound: 0.0395456
time: 6.60 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1074390, 0.1074012
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0938909, 0.0939399
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1765893, 0.1764674
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2552187, 0.2552119
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2804402, 0.2803135
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3036165, 0.3036054
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3028462, 0.3028464
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8708397, 0.8707548
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2802457, 0.2802649
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0738200, 0.0738287

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3512

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2657

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395358, upper bound: 0.0396023
time: 4.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395360, upper bound: 0.0395424
time: 43.63 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 53.95 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 53.95
Output dim: 1, lower bound: -0.0395936, upper bound: 0.0395471
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 53.95
Output dim: 1, lower bound: -0.0395930, upper bound: 0.0395456
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 53.95
Output dim: 1, lower bound: -0.0395358, upper bound: 0.0396023
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 53.95
Output dim: 1, lower bound: -0.0395360, upper bound: 0.0395424

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1071495, 0.1071138
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0934631, 0.0935090
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1754667, 0.1753519
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2550944, 0.2550808
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2792978, 0.2791796
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3033298, 0.3033088
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3028548, 0.3028535
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8702713, 0.8701889
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2801191, 0.2801370
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0737851, 0.0737926

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3512

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2206

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395347, upper bound: 0.0395893
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395220, upper bound: 0.0396018
time: 4.23 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 14.25 seconds
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 14.25
Output dim: 1, lower bound: -0.0395347, upper bound: 0.0395893
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 14.25
Output dim: 1, lower bound: -0.0395220, upper bound: 0.0396018

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 27.78 + 477.01 = 504.80 seconds
