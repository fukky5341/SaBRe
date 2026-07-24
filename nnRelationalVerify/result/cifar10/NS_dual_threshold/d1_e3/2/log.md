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
execution time: IAR + RelationalAnalysis = 7.73 + 19.41 = 27.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0396317, upper bound: 0.0396418

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 300
type: B, layer: 1, pos: 300
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3125
type: B, layer: 1, pos: 3125
type: A, layer: 1, pos: 382
type: B, layer: 1, pos: 382
type: A, layer: 1, pos: 2434
type: B, layer: 1, pos: 2434
type: A, layer: 1, pos: 310
type: B, layer: 1, pos: 310
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3129
type: B, layer: 1, pos: 3129
type: A, layer: 1, pos: 2659
type: B, layer: 1, pos: 2659
type: A, layer: 1, pos: 2970
type: B, layer: 1, pos: 2970
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3446
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3126
type: B, layer: 1, pos: 3126
type: A, layer: 1, pos: 427
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 2451
type: B, layer: 1, pos: 2451
type: A, layer: 1, pos: 2542
type: B, layer: 1, pos: 2542
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2657
type: B, layer: 1, pos: 2657
type: A, layer: 1, pos: 2437
type: B, layer: 1, pos: 2437
type: A, layer: 1, pos: 3219
type: B, layer: 1, pos: 3219
type: A, layer: 1, pos: 2569
type: B, layer: 1, pos: 2569
type: A, layer: 1, pos: 3233
type: B, layer: 1, pos: 3233
type: A, layer: 1, pos: 3108
type: B, layer: 1, pos: 3108
type: A, layer: 1, pos: 2988
type: B, layer: 1, pos: 2988
type: A, layer: 1, pos: 2554
type: B, layer: 1, pos: 2554
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 3440
type: B, layer: 1, pos: 3440
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2036
type: B, layer: 1, pos: 2036
type: A, layer: 1, pos: 2515
type: B, layer: 1, pos: 2515
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2494
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 2378
type: B, layer: 1, pos: 2378
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 2680
type: B, layer: 1, pos: 2680
type: A, layer: 1, pos: 2131
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 3061
type: A, layer: 1, pos: 3061
type: B, layer: 1, pos: 2370
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2115
type: B, layer: 1, pos: 2115
type: A, layer: 1, pos: 2094
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2221
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2990
type: B, layer: 1, pos: 2990
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2966
type: B, layer: 1, pos: 2966
type: A, layer: 1, pos: 3512
type: B, layer: 1, pos: 3512
type: A, layer: 1, pos: 3295
type: B, layer: 1, pos: 3295
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 3235
type: B, layer: 1, pos: 3235
type: A, layer: 1, pos: 3334
type: B, layer: 1, pos: 3334
type: A, layer: 1, pos: 2989
type: B, layer: 1, pos: 2989
type: A, layer: 1, pos: 2583
type: B, layer: 1, pos: 2583
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2962
type: A, layer: 1, pos: 2962
type: B, layer: 1, pos: 3062
type: A, layer: 1, pos: 3062
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2977
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3530
type: B, layer: 1, pos: 3530
type: A, layer: 1, pos: 3528
type: B, layer: 1, pos: 3528
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3027
type: B, layer: 1, pos: 3027
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3236
type: B, layer: 1, pos: 3236
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 3323
type: B, layer: 1, pos: 3323
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 2968
type: B, layer: 1, pos: 2968
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 3439
type: B, layer: 1, pos: 3439
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2357
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 2997
type: B, layer: 1, pos: 2997
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 2132
type: B, layer: 1, pos: 2132
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 3461
type: B, layer: 1, pos: 3461
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
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
type: A, layer: 1, pos: 300

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396271, upper bound: 0.0393467
time: 6.64 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0396320, upper bound: 0.0396409
time: 3.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.39 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.39
Output dim: 1, lower bound: -0.0396271, upper bound: 0.0393467
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.39
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

Time for backsubstitution: 5.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3125
type: A, layer: 1, pos: 3125
type: B, layer: 1, pos: 382
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 2434
type: B, layer: 1, pos: 2434
type: A, layer: 1, pos: 310
type: B, layer: 1, pos: 310
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3129
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2659
type: B, layer: 1, pos: 2659
type: A, layer: 1, pos: 2970
type: B, layer: 1, pos: 2970
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3446
type: A, layer: 1, pos: 3446
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 2417
type: B, layer: 1, pos: 3126
type: A, layer: 1, pos: 3126
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 427
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 2451
type: B, layer: 1, pos: 2451
type: A, layer: 1, pos: 2542
type: B, layer: 1, pos: 2542
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2657
type: A, layer: 1, pos: 2657
type: B, layer: 1, pos: 2437
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3219
type: B, layer: 1, pos: 3219
type: A, layer: 1, pos: 2569
type: B, layer: 1, pos: 2569
type: A, layer: 1, pos: 3233
type: B, layer: 1, pos: 3233
type: A, layer: 1, pos: 3108
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2988
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2554
type: B, layer: 1, pos: 2554
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 3440
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2036
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2515
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2494
type: A, layer: 1, pos: 2494
type: B, layer: 1, pos: 300
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 2378
type: B, layer: 1, pos: 2378
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 2680
type: B, layer: 1, pos: 2680
type: A, layer: 1, pos: 2131
type: B, layer: 1, pos: 2131
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 3061
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2370
type: B, layer: 1, pos: 2370
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2115
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2094
type: B, layer: 1, pos: 2094
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2221
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2990
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2966
type: B, layer: 1, pos: 3512
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 3295
type: B, layer: 1, pos: 3295
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2045
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3235
type: A, layer: 1, pos: 3235
type: A, layer: 1, pos: 3334
type: B, layer: 1, pos: 3334
type: B, layer: 1, pos: 2989
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2583
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2962
type: A, layer: 1, pos: 2962
type: B, layer: 1, pos: 3062
type: A, layer: 1, pos: 3062
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2977
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 3530
type: B, layer: 1, pos: 3530
type: B, layer: 1, pos: 3528
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3027
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3236
type: B, layer: 1, pos: 3236
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 3323
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 2968
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 3439
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2357
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 2132
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 3461
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
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
type: A, layer: 1, pos: 2297

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395980, upper bound: 0.0393134
time: 3.88 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395929, upper bound: 0.0393059
time: 50.00 seconds

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

Time for backsubstitution: 5.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3125
type: B, layer: 1, pos: 3125
type: A, layer: 1, pos: 382
type: B, layer: 1, pos: 382
type: B, layer: 1, pos: 2434
type: A, layer: 1, pos: 2434
type: B, layer: 1, pos: 310
type: A, layer: 1, pos: 310
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3129
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2659
type: A, layer: 1, pos: 2659
type: B, layer: 1, pos: 2970
type: A, layer: 1, pos: 2970
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3446
type: B, layer: 1, pos: 3446
type: A, layer: 1, pos: 2417
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 3126
type: B, layer: 1, pos: 3126
type: A, layer: 1, pos: 427
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 2451
type: A, layer: 1, pos: 2451
type: B, layer: 1, pos: 2542
type: A, layer: 1, pos: 2542
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2657
type: B, layer: 1, pos: 2657
type: A, layer: 1, pos: 2437
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 3219
type: A, layer: 1, pos: 3219
type: B, layer: 1, pos: 2569
type: A, layer: 1, pos: 2569
type: B, layer: 1, pos: 3233
type: A, layer: 1, pos: 3233
type: B, layer: 1, pos: 3108
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2988
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2554
type: A, layer: 1, pos: 2554
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 3440
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2036
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2515
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 300
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 3092
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2494
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 2378
type: A, layer: 1, pos: 2378
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 2680
type: A, layer: 1, pos: 2680
type: B, layer: 1, pos: 2131
type: A, layer: 1, pos: 2131
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 3061
type: B, layer: 1, pos: 2370
type: A, layer: 1, pos: 2370
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2115
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2094
type: A, layer: 1, pos: 2094
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: A, layer: 1, pos: 2990
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 3512
type: B, layer: 1, pos: 3512
type: B, layer: 1, pos: 3295
type: A, layer: 1, pos: 3295
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 2966
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3235
type: B, layer: 1, pos: 3235
type: B, layer: 1, pos: 3334
type: A, layer: 1, pos: 3334
type: A, layer: 1, pos: 2989
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2583
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2962
type: B, layer: 1, pos: 2962
type: A, layer: 1, pos: 3062
type: B, layer: 1, pos: 3062
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 2977
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 3530
type: A, layer: 1, pos: 3530
type: A, layer: 1, pos: 3528
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3027
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3236
type: A, layer: 1, pos: 3236
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2968
type: B, layer: 1, pos: 3323
type: A, layer: 1, pos: 3323
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 3439
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2357
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 2997
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 2132
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 3461
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
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
type: B, layer: 1, pos: 2297

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395977, upper bound: 0.0396136
time: 4.52 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0395984, upper bound: 0.0396029
time: 19.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.82 seconds
NS_A1_A1, status: Status.VERIFIED, split count: 2, time: 29.82
Output dim: 1, lower bound: -0.0395980, upper bound: 0.0393134
NS_A1_A2, status: Status.VERIFIED, split count: 2, time: 29.82
Output dim: 1, lower bound: -0.0395929, upper bound: 0.0393059
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.82
Output dim: 1, lower bound: -0.0395977, upper bound: 0.0396136
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.82
Output dim: 1, lower bound: -0.0395984, upper bound: 0.0396029

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.7149252, -0.2501926, -0.7149244, -0.2502021, -0.1071522, 0.1065415
1: 0.2782388, 0.7803826, 0.2783406, 0.7803825, -0.0932835, 0.0857557
2: -4.7610617, -3.8633909, -4.7610474, -3.8633833, -0.1766472, 0.1765050
3: -6.3806767, -5.1745052, -6.3806458, -5.1745057, -0.2550827, 0.2538203
4: -6.3705125, -5.1030273, -6.3705111, -5.1030717, -0.2798530, 0.2751295
5: -6.5109172, -5.2259088, -6.5108995, -5.2259092, -0.3009878, 0.3034588
6: -8.7728710, -7.5823894, -8.7727976, -7.5823889, -0.3017448, 0.2988998
7: -4.3565865, -2.4834423, -4.3565831, -2.4834878, -0.8671582, 0.8708192
8: -0.0788859, 0.5000726, -0.0788902, 0.5000652, -0.2784577, 0.2792405
9: 0.8221922, 1.1012890, 0.8222044, 1.1012890, -0.0737786, 0.0733549

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3125
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 382
type: A, layer: 1, pos: 382
type: B, layer: 1, pos: 2434
type: A, layer: 1, pos: 2434
type: B, layer: 1, pos: 310
type: A, layer: 1, pos: 310
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3129
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2659
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2970
type: B, layer: 1, pos: 2970
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3446
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3126
type: B, layer: 1, pos: 3126
type: A, layer: 1, pos: 427
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 2451
type: A, layer: 1, pos: 2451
type: B, layer: 1, pos: 2542
type: A, layer: 1, pos: 2542
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 2657
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2437
type: B, layer: 1, pos: 2437
type: A, layer: 1, pos: 3219
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 2569
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3233
type: B, layer: 1, pos: 3233
type: A, layer: 1, pos: 3108
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2988
type: A, layer: 1, pos: 2988
type: B, layer: 1, pos: 2554
type: A, layer: 1, pos: 2554
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 3440
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2036
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2515
type: A, layer: 1, pos: 2515
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 300
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2494
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2378
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2680
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2131
type: B, layer: 1, pos: 2131
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 3061
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2370
type: B, layer: 1, pos: 2370
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2115
type: A, layer: 1, pos: 2115
type: B, layer: 1, pos: 2094
type: A, layer: 1, pos: 2094
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2221
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 2990
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2966
type: B, layer: 1, pos: 3512
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 3295
type: B, layer: 1, pos: 3295
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2045
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 3235
type: B, layer: 1, pos: 3235
type: B, layer: 1, pos: 3334
type: A, layer: 1, pos: 3334
type: A, layer: 1, pos: 2989
type: B, layer: 1, pos: 2989
type: A, layer: 1, pos: 2583
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2962
type: A, layer: 1, pos: 2962
type: B, layer: 1, pos: 3062
type: A, layer: 1, pos: 3062
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2977
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 3530
type: A, layer: 1, pos: 3530
type: B, layer: 1, pos: 3528
type: A, layer: 1, pos: 3528
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3027
type: B, layer: 1, pos: 3027
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3236
type: A, layer: 1, pos: 3236
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 3323
type: A, layer: 1, pos: 3323
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 3439
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2357
type: B, layer: 1, pos: 2357
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2997
type: A, layer: 1, pos: 2997
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 2132
type: B, layer: 1, pos: 2132
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 3461
type: B, layer: 1, pos: 3461
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
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

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3125

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395587, upper bound: 0.0395629
time: 194.06 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395604, upper bound: 0.0392725
time: 100.46 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.7149172, -0.2503365, -0.7150296, -0.2503683, -0.1071487, 0.1069040
1: 0.2788191, 0.7803826, 0.2789317, 0.7805887, -0.0937770, 0.0853910
2: -4.7610159, -3.8633933, -4.7609973, -3.8631420, -0.1771299, 0.1765106
3: -6.3806252, -5.1745090, -6.3806047, -5.1740584, -0.2558529, 0.2538249
4: -6.3705015, -5.1031642, -6.3710327, -5.1032019, -0.2797935, 0.2758070
5: -6.5108767, -5.2259140, -6.5108547, -5.2256980, -0.3012066, 0.3035696
6: -8.7727947, -7.5823903, -8.7727318, -7.5813522, -0.3036020, 0.2988948
7: -4.3565788, -2.4834814, -4.3574047, -2.4835200, -0.8670868, 0.8714713
8: -0.0788764, 0.4999754, -0.0789304, 0.4999508, -0.2784581, 0.2794047
9: 0.8222626, 1.1012881, 0.8222749, 1.1013007, -0.0737547, 0.0733048

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3125
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 382
type: A, layer: 1, pos: 382
type: B, layer: 1, pos: 2434
type: A, layer: 1, pos: 2434
type: B, layer: 1, pos: 310
type: A, layer: 1, pos: 310
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3129
type: B, layer: 1, pos: 3129
type: A, layer: 1, pos: 2970
type: B, layer: 1, pos: 2659
type: A, layer: 1, pos: 2659
type: B, layer: 1, pos: 2970
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3446
type: A, layer: 1, pos: 3446
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3126
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 427
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 2451
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2542
type: B, layer: 1, pos: 2542
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2657
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2437
type: B, layer: 1, pos: 2437
type: A, layer: 1, pos: 3219
type: B, layer: 1, pos: 3219
type: A, layer: 1, pos: 2569
type: B, layer: 1, pos: 2569
type: A, layer: 1, pos: 3233
type: B, layer: 1, pos: 3233
type: A, layer: 1, pos: 3108
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2988
type: A, layer: 1, pos: 2988
type: B, layer: 1, pos: 2554
type: A, layer: 1, pos: 2554
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 2515
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 3440
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2036
type: A, layer: 1, pos: 2515
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 300
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 3092
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 2378
type: B, layer: 1, pos: 2378
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2680
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2131
type: B, layer: 1, pos: 3061
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2370
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2966
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2094
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2968
type: A, layer: 1, pos: 2990
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 3512
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 3295
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2045
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 3235
type: B, layer: 1, pos: 3235
type: A, layer: 1, pos: 3334
type: B, layer: 1, pos: 3334
type: A, layer: 1, pos: 2989
type: B, layer: 1, pos: 2989
type: A, layer: 1, pos: 2583
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2962
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2962
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 3530
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 3530
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3528
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3027
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3236
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 3323
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 3439
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2357
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3461
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 158
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
type: A, layer: 1, pos: 2966
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 2968

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3125

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395591, upper bound: 0.0395624
time: 4.07 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0395590, upper bound: 0.0392713
time: 24.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 34.45 seconds
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 34.45
Output dim: 1, lower bound: -0.0395587, upper bound: 0.0395629
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 34.45
Output dim: 1, lower bound: -0.0395604, upper bound: 0.0392725
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 34.45
Output dim: 1, lower bound: -0.0395591, upper bound: 0.0395624
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 34.45
Output dim: 1, lower bound: -0.0395590, upper bound: 0.0392713

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 27.14 + 434.78 = 461.92 seconds
