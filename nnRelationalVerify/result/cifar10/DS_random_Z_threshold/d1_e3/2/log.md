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
execution time: IAR + RelationalAnalysis = 8.26 + 19.65 = 27.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0396317, upper bound: 0.0396418

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3437

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 751

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396298, upper bound: 0.0396382
time: 17.67 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396293, upper bound: 0.0396337
time: 74.91 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 92.59 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 92.59
Output dim: 1, lower bound: -0.0396298, upper bound: 0.0396382
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 92.59
Output dim: 1, lower bound: -0.0396293, upper bound: 0.0396337

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1074679, 0.1074656
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0940038, 0.0940036
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1767089, 0.1767092
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2552854, 0.2552854
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2805206, 0.2805175
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3037086, 0.3037086
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3028479, 0.3028486
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8710366, 0.8710369
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2802597, 0.2802594
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0738474, 0.0738474

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3439

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2136

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396261, upper bound: 0.0395790
time: 9.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395726, upper bound: 0.0396366
time: 3.76 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1074656, 0.1074679
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0940036, 0.0940038
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1767092, 0.1767089
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2552854, 0.2552854
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2805175, 0.2805206
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3037086, 0.3037086
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3028486, 0.3028479
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8710369, 0.8710366
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2802594, 0.2802596
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0738474, 0.0738474

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3306

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2652

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396231, upper bound: 0.0396310
time: 5.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396231, upper bound: 0.0396300
time: 36.78 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 47.91 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 47.91
Output dim: 1, lower bound: -0.0396261, upper bound: 0.0395790
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 47.91
Output dim: 1, lower bound: -0.0395726, upper bound: 0.0396366
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 47.91
Output dim: 1, lower bound: -0.0396231, upper bound: 0.0396310
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 47.91
Output dim: 1, lower bound: -0.0396231, upper bound: 0.0396300

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1065210, 0.1064953
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0923675, 0.0923368
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1764596, 0.1764593
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2474019, 0.2476383
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2775168, 0.2775964
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2957854, 0.2960217
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2933978, 0.2936784
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8672644, 0.8673825
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2794672, 0.2794745
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0724423, 0.0724548

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 750

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396214, upper bound: 0.0395698
time: 27.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396204, upper bound: 0.0395791
time: 4.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1064977, 0.1065187
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0923371, 0.0923672
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1764590, 0.1764598
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2476383, 0.2474020
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2775995, 0.2775138
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2960216, 0.2957854
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2936776, 0.2933985
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8673822, 0.8672647
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2794747, 0.2794670
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0724547, 0.0724423

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2378

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2387

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395676, upper bound: 0.0396167
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395536, upper bound: 0.0396261
time: 70.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1074113, 0.1074112
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0939418, 0.0939421
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1760879, 0.1760597
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2548119, 0.2548066
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2797605, 0.2797381
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3033629, 0.3033587
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3027350, 0.3027291
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8706075, 0.8705909
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2801872, 0.2801943
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0738261, 0.0738251

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 698

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396232, upper bound: 0.0396265
time: 198.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396238, upper bound: 0.0396309
time: 4.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1074089, 0.1074136
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0939418, 0.0939420
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1760600, 0.1760876
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2548065, 0.2548118
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2797350, 0.2797635
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3033587, 0.3033629
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3027298, 0.3027342
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8705910, 0.8706072
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2801940, 0.2801875
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0738251, 0.0738260

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2297

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2370

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396210, upper bound: 0.0396266
time: 36.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396238, upper bound: 0.0396289
time: 14.60 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 57.71 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 57.71
Output dim: 1, lower bound: -0.0396214, upper bound: 0.0395698
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 57.71
Output dim: 1, lower bound: -0.0396204, upper bound: 0.0395791
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 57.71
Output dim: 1, lower bound: -0.0395676, upper bound: 0.0396167
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 57.71
Output dim: 1, lower bound: -0.0395536, upper bound: 0.0396261
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 57.71
Output dim: 1, lower bound: -0.0396232, upper bound: 0.0396265
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 57.71
Output dim: 1, lower bound: -0.0396238, upper bound: 0.0396309
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 57.71
Output dim: 1, lower bound: -0.0396210, upper bound: 0.0396266
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 57.71
Output dim: 1, lower bound: -0.0396238, upper bound: 0.0396289

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1064360, 0.1064105
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0922352, 0.0922043
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1748382, 0.1748397
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2436265, 0.2438152
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2760767, 0.2761578
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2924091, 0.2925986
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2913711, 0.2916089
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8663937, 0.8665155
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2794383, 0.2794460
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0723763, 0.0723869

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 679

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396215, upper bound: 0.0395779
time: 4.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396215, upper bound: 0.0395781
time: 4.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1064362, 0.1064103
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0922350, 0.0922045
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1748400, 0.1748380
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2435789, 0.2438629
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2760782, 0.2761562
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2923623, 0.2926454
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2913284, 0.2916517
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8663975, 0.8665118
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2794387, 0.2794455
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0723744, 0.0723888

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2968

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 149

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396204, upper bound: 0.0395785
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396204, upper bound: 0.0395784
time: 4.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1064653, 0.1064867
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0920851, 0.0921090
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1752405, 0.1752786
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2402760, 0.2401053
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2759735, 0.2759146
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2881694, 0.2880118
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2894059, 0.2891447
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8622073, 0.8621939
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2794270, 0.2794174
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0723973, 0.0723827

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2515

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3446

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395666, upper bound: 0.0395099
time: 111.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395203, upper bound: 0.0395086
time: 66.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1064657, 0.1064863
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0920789, 0.0921152
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1752778, 0.1752412
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2403416, 0.2400397
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2760004, 0.2758878
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2882482, 0.2879331
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2894239, 0.2891268
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8623115, 0.8620896
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2794251, 0.2794193
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0723951, 0.0723849

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 678

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2992

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395466, upper bound: 0.0396101
time: 4.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395345, upper bound: 0.0396237
time: 4.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1074106, 0.1074104
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0939418, 0.0939421
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1760790, 0.1760508
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2548050, 0.2547999
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2797271, 0.2797043
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3033558, 0.3033513
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3027182, 0.3027121
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8706074, 0.8705908
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2801841, 0.2801911
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0738233, 0.0738223

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3339

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395898, upper bound: 0.0395926
time: 26.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395898, upper bound: 0.0395931
time: 22.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1074105, 0.1074105
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0939418, 0.0939421
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1760790, 0.1760507
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2548052, 0.2547997
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2797266, 0.2797047
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3033556, 0.3033514
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3027179, 0.3027123
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8706074, 0.8705908
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2801841, 0.2801912
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0738233, 0.0738223

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2990

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396200, upper bound: 0.0396090
time: 92.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396067, upper bound: 0.0396275
time: 6.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1073966, 0.1074014
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0939418, 0.0939420
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1759046, 0.1759355
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2547403, 0.2547474
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2796989, 0.2797285
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3031909, 0.3031957
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3014947, 0.3015219
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8703101, 0.8703324
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2801189, 0.2801113
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0738154, 0.0738162

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2992

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2568

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396001, upper bound: 0.0396087
time: 105.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396027, upper bound: 0.0396083
time: 51.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1073968, 0.1074013
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0939418, 0.0939420
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1759078, 0.1759322
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2547421, 0.2547456
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2797000, 0.2797274
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3031914, 0.3031952
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3015175, 0.3014991
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8703163, 0.8703262
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2801179, 0.2801123
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0738153, 0.0738163

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2357

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3440

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396215, upper bound: 0.0395940
time: 4.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395860, upper bound: 0.0396227
time: 189.27 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 199.92 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0396215, upper bound: 0.0395779
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0396215, upper bound: 0.0395781
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0396204, upper bound: 0.0395785
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0396204, upper bound: 0.0395784
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0395666, upper bound: 0.0395099
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0395203, upper bound: 0.0395086
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0395466, upper bound: 0.0396101
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0395345, upper bound: 0.0396237
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0395898, upper bound: 0.0395926
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0395898, upper bound: 0.0395931
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0396200, upper bound: 0.0396090
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0396067, upper bound: 0.0396275
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0396001, upper bound: 0.0396087
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0396027, upper bound: 0.0396083
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0396215, upper bound: 0.0395940
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 199.92
Output dim: 1, lower bound: -0.0395860, upper bound: 0.0396227

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1064360, 0.1064105
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0922352, 0.0922043
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1748382, 0.1748397
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2436265, 0.2438152
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2760767, 0.2761578
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2924091, 0.2925986
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2913711, 0.2916089
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8663937, 0.8665155
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2794383, 0.2794460
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0723763, 0.0723869

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 839

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2473

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396225, upper bound: 0.0395724
time: 38.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396225, upper bound: 0.0395754
time: 38.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1064360, 0.1064105
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0922352, 0.0922043
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1748382, 0.1748397
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2436265, 0.2438152
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2760767, 0.2761578
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2924091, 0.2925986
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2913711, 0.2916089
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8663937, 0.8665155
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2794383, 0.2794460
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0723763, 0.0723869

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 606

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396158, upper bound: 0.0395694
time: 4.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396129, upper bound: 0.0395726
time: 4.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1064362, 0.1064103
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0922350, 0.0922045
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1748400, 0.1748380
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2435789, 0.2438629
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2760782, 0.2761562
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2923623, 0.2926454
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2913284, 0.2916517
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8663975, 0.8665118
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2794387, 0.2794455
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0723744, 0.0723888

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2997

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2046

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396192, upper bound: 0.0395785
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396198, upper bound: 0.0395770
time: 4.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1064362, 0.1064103
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0922350, 0.0922045
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1748400, 0.1748380
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2435789, 0.2438629
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2760782, 0.2761562
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2923623, 0.2926454
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2913284, 0.2916517
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8663975, 0.8665118
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2794387, 0.2794455
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0723744, 0.0723888

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2357

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2698

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396200, upper bound: 0.0395728
time: 43.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396200, upper bound: 0.0395698
time: 47.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1063582, 0.1063785
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0913649, 0.0913973
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1750893, 0.1750493
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2384020, 0.2381232
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2760052, 0.2758926
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2867078, 0.2864096
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2863958, 0.2861311
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8619394, 0.8617113
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2792485, 0.2792517
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0718900, 0.0718780

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 824

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 750

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395452, upper bound: 0.0395497
time: 214.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395457, upper bound: 0.0396094
time: 4.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1063579, 0.1063788
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0913610, 0.0914012
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1750858, 0.1750528
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2384252, 0.2381000
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2760051, 0.2758926
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.2867247, 0.2863927
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.2864282, 0.2860987
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8619330, 0.8617176
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2792575, 0.2792427
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0718882, 0.0718798

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2997

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395334, upper bound: 0.0396114
time: 8.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395248, upper bound: 0.0396177
time: 13.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7149288, -0.2501314, -0.7149288, -0.2501314, -0.1073434, 0.1073431
1: 0.2777508, 0.7803833, 0.2777508, 0.7803833, -0.0938290, 0.0938288
2: -4.7611332, -3.8633087, -4.7611332, -3.8633087, -0.1760403, 0.1760118
3: -6.3808222, -5.1745038, -6.3808222, -5.1745038, -0.2541544, 0.2541229
4: -6.3705168, -5.1027236, -6.3705168, -5.1027236, -0.2796287, 0.2796093
5: -6.5110073, -5.2259064, -6.5110073, -5.2259064, -0.3027995, 0.3027644
6: -8.7732239, -7.5823846, -8.7732239, -7.5823846, -0.3020065, 0.3020060
7: -4.3566046, -2.4832237, -4.3566046, -2.4832237, -0.8704293, 0.8704011
8: -0.0789319, 0.5001076, -0.0789319, 0.5001076, -0.2800859, 0.2800986
9: 0.8221340, 1.1012899, 0.8221340, 1.1012899, -0.0736439, 0.0736322

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 635
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2136

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396151, upper bound: 0.0395583
time: 10.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395632, upper bound: 0.0396106
time: 147.66 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 163.87 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0396225, upper bound: 0.0395724
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0396225, upper bound: 0.0395754
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0396158, upper bound: 0.0395694
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0396129, upper bound: 0.0395726
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0396192, upper bound: 0.0395785
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0396198, upper bound: 0.0395770
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0396200, upper bound: 0.0395728
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0396200, upper bound: 0.0395698
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0395452, upper bound: 0.0395497
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0395457, upper bound: 0.0396094
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0395334, upper bound: 0.0396114
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0395248, upper bound: 0.0396177
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0396151, upper bound: 0.0395583
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 163.87
Output dim: 1, lower bound: -0.0395632, upper bound: 0.0396106
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 163.87
Output dim: 1, lower bound: -0.0396067, upper bound: 0.0396275
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 163.87
Output dim: 1, lower bound: -0.0396001, upper bound: 0.0396087
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 163.87
Output dim: 1, lower bound: -0.0396027, upper bound: 0.0396083
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 163.87
Output dim: 1, lower bound: -0.0396215, upper bound: 0.0395940
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 163.87
Output dim: 1, lower bound: -0.0395860, upper bound: 0.0396227

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 27.91 + 1924.36 = 1952.27 seconds
