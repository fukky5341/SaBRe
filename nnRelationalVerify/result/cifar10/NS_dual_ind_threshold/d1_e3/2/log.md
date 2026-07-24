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
execution time: IAR + RelationalAnalysis = 7.86 + 19.99 = 27.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0396317, upper bound: 0.0396418

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 300
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 310
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3235
type: A, layer: 1, pos: 3334
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3530
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3119
type: A, layer: 1, pos: 3138

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 300

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396271, upper bound: 0.0393467
time: 6.85 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396320, upper bound: 0.0396409
time: 3.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.71 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.71
Output dim: 1, lower bound: -0.0396271, upper bound: 0.0393467
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.71
Output dim: 1, lower bound: -0.0396320, upper bound: 0.0396409

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.7140996, -0.2510068, -0.7146971, -0.2508931, -0.1058621, 0.1061724
1: 0.2811891, 0.7757887, 0.2778023, 0.7763698, -0.0867870, 0.0893329
2: -4.7608652, -3.8637011, -4.7609372, -3.8636637, -0.1757469, 0.1760779
3: -6.3799577, -5.1758800, -6.3807516, -5.1756430, -0.2531947, 0.2538294
4: -6.3672771, -5.1066999, -6.3700676, -5.1061945, -0.2739436, 0.2758704
5: -6.5089641, -5.2274361, -6.5093603, -5.2260709, -0.3013613, 0.3005384
6: -8.7711201, -7.5859065, -8.7732019, -7.5854235, -0.2982058, 0.2996307
7: -4.3490939, -2.4882143, -4.3501363, -2.4836211, -0.8633294, 0.8596557
8: -0.0749916, 0.4990016, -0.0759469, 0.5000980, -0.2759223, 0.2755499
9: 0.8224773, 1.1008211, 0.8221767, 1.1008871, -0.0730498, 0.0733212

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 382
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 310
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 300
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3233
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3235
type: B, layer: 1, pos: 3334
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 3530
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 3512
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2966
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 389
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3119
type: B, layer: 1, pos: 3138

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 382

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395387, upper bound: 0.0393317
time: 201.69 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396204, upper bound: 0.0393349
time: 40.43 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.7149285, -0.2501411, -0.7149287, -0.2501398, -0.1072930, 0.1066613
1: 0.2777519, 0.7803829, 0.2777518, 0.7803830, -0.0939871, 0.0866132
2: -4.7611332, -3.8633866, -4.7611332, -3.8633780, -0.1767664, 0.1766411
3: -6.3808222, -5.1745038, -6.3808222, -5.1745033, -0.2552856, 0.2540544
4: -6.3705168, -5.1027670, -6.3705177, -5.1027622, -0.2802168, 0.2754388
5: -6.5110054, -5.2259068, -6.5110054, -5.2259068, -0.3011035, 0.3035825
6: -8.7732239, -7.5823874, -8.7732239, -7.5823874, -0.3021702, 0.2994008
7: -4.3566017, -2.4832239, -4.3566022, -2.4832239, -0.8674155, 0.8710387
8: -0.0788907, 0.5001076, -0.0788960, 0.5001076, -0.2785395, 0.2793129
9: 0.8221341, 1.1012897, 0.8221340, 1.1012897, -0.0738529, 0.0734444

Time for backsubstitution: 6.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 382
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 310
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 300
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3233
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3235
type: B, layer: 1, pos: 3334
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 3530
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 3512
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2966
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 389
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3119
type: B, layer: 1, pos: 3138

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 382

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395402, upper bound: 0.0396298
time: 6.14 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396258, upper bound: 0.0396355
time: 6.21 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 18.51 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 18.51
Output dim: 1, lower bound: -0.0395387, upper bound: 0.0393317
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.51
Output dim: 1, lower bound: -0.0396204, upper bound: 0.0393349
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.51
Output dim: 1, lower bound: -0.0395402, upper bound: 0.0396298
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.51
Output dim: 1, lower bound: -0.0396258, upper bound: 0.0396355

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.7140908, -0.2510067, -0.7147130, -0.2472544, -0.1094901, 0.1049722
1: 0.2811906, 0.7756119, 0.2758126, 0.7761863, -0.0857628, 0.0917106
2: -4.7608075, -3.8637011, -4.7609463, -3.8612230, -0.1775902, 0.1755288
3: -6.3799558, -5.1767535, -6.3843799, -5.1765275, -0.2510795, 0.2589889
4: -6.3672209, -5.1067009, -6.3700376, -5.1020713, -0.2777184, 0.2745547
5: -6.5089631, -5.2287664, -6.5120211, -5.2274585, -0.2996157, 0.3053732
6: -8.7711163, -7.5868282, -8.7748270, -7.5863791, -0.2977840, 0.3002707
7: -4.3490024, -2.4882159, -4.3501124, -2.4766896, -0.8696012, 0.8565766
8: -0.0749872, 0.4989942, -0.0772399, 0.5000968, -0.2757568, 0.2768377
9: 0.8224787, 1.1008098, 0.8202932, 1.1008805, -0.0722485, 0.0749511

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 310
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3235
type: A, layer: 1, pos: 3334
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3530
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3119
type: A, layer: 1, pos: 3138

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3125

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395831, upper bound: 0.0392891
time: 54.77 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395827, upper bound: 0.0393026
time: 4.33 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.7136250, -0.2501483, -0.7133897, -0.2501483, -0.1060431, 0.1052502
1: 0.2780215, 0.7797558, 0.2780701, 0.7796640, -0.0924647, 0.0851656
2: -4.7602000, -3.8633928, -4.7600307, -3.8633859, -0.1759403, 0.1757508
3: -6.3806882, -5.1757832, -6.3806639, -5.1760144, -0.2527803, 0.2518764
4: -6.3690119, -5.1028500, -6.3687401, -5.1028581, -0.2787459, 0.2738043
5: -6.5109191, -5.2269731, -6.5109048, -5.2271657, -0.2990026, 0.3017646
6: -8.7728453, -7.5833716, -8.7727757, -7.5835156, -0.3001195, 0.2972760
7: -4.3539357, -2.4833455, -4.3534570, -2.4833670, -0.8639838, 0.8671553
8: -0.0788432, 0.4994911, -0.0788400, 0.4993798, -0.2777884, 0.2786529
9: 0.8222921, 1.1004984, 0.8223207, 1.1003653, -0.0726703, 0.0723628

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 310
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3235
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3334
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3530
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3119
type: A, layer: 1, pos: 3138

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3125

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395019, upper bound: 0.0395846
time: 61.83 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395018, upper bound: 0.0395922
time: 5.04 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.7149194, -0.2501412, -0.7149448, -0.2465010, -0.1109209, 0.1054624
1: 0.2777534, 0.7802061, 0.2757620, 0.7801994, -0.0929627, 0.0889888
2: -4.7610750, -3.8633869, -4.7611442, -3.8609374, -0.1786103, 0.1760898
3: -6.3808203, -5.1753783, -6.3844490, -5.1753883, -0.2531702, 0.2592122
4: -6.3704610, -5.1027679, -6.3704882, -5.0986385, -0.2839915, 0.2741215
5: -6.5110040, -5.2272372, -6.5136647, -5.2272949, -0.2993578, 0.3084173
6: -8.7732191, -7.5833077, -8.7748470, -7.5833421, -0.3017381, 0.3000436
7: -4.3565106, -2.4832251, -4.3565788, -2.4762924, -0.8736690, 0.8679606
8: -0.0788864, 0.5001000, -0.0801846, 0.5001063, -0.2783737, 0.2805922
9: 0.8221354, 1.1012782, 0.8202508, 1.1012831, -0.0730516, 0.0750728

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 310
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3235
type: A, layer: 1, pos: 3334
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3530
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 389
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3119
type: A, layer: 1, pos: 3138

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3125

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395873, upper bound: 0.0395896
time: 5.81 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395874, upper bound: 0.0395901
time: 19.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.55 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 31.55
Output dim: 1, lower bound: -0.0395831, upper bound: 0.0392891
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 31.55
Output dim: 1, lower bound: -0.0395827, upper bound: 0.0393026
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 31.55
Output dim: 1, lower bound: -0.0395019, upper bound: 0.0395846
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 31.55
Output dim: 1, lower bound: -0.0395018, upper bound: 0.0395922
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 31.55
Output dim: 1, lower bound: -0.0395873, upper bound: 0.0395896
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 31.55
Output dim: 1, lower bound: -0.0395874, upper bound: 0.0395901

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 27.85 + 447.16 = 475.01 seconds
