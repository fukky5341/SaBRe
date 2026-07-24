## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 10)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0125345529


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6555245, 0.6555245)
1: (-4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7175561, 0.7175560)
2: (-0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0575615, 0.0575615)
3: (-1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2591920, 0.2591919)
4: (0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0990137, 0.0990137)
5: (-1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2356481, 0.2356481)
6: (0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0748282, 0.0748282)
7: (-0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2572221, 0.2572222)
8: (-5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6054912, 0.6054909)
9: (-3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5246348, 0.5246348)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.88 + 23.22 = 31.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0125471, upper bound: 0.0125478

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 503
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2720
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2786
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2788
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 2730
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2772
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2792
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2880
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2804
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2587

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125394, upper bound: 0.0125270
time: 152.03 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125395, upper bound: 0.0125403
time: 8.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 161.02 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 161.02
Output dim: 4, lower bound: -0.0125394, upper bound: 0.0125270
NS_A2, status: Status.UNKNOWN, split count: 1, time: 161.02
Output dim: 4, lower bound: -0.0125395, upper bound: 0.0125403

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.6976151, -2.2994561, -3.6980033, -2.2994566, -0.6507485, 0.6511276
1: -4.0253515, -2.1158934, -4.0259962, -2.1158934, -0.7094260, 0.7101388
2: -0.4317880, -0.2510268, -0.4317884, -0.2510137, -0.0574246, 0.0574142
3: -1.7299957, -1.2387537, -1.7299963, -1.2386277, -0.2578477, 0.2577428
4: 0.0898123, 0.2676256, 0.0898122, 0.2676523, -0.0987653, 0.0987475
5: -1.5401621, -1.0447066, -1.5401627, -1.0445633, -0.2340087, 0.2338681
6: 0.0139760, 0.2834425, 0.0139760, 0.2835371, -0.0738978, 0.0738122
7: -0.8505571, -0.4270343, -0.8505584, -0.4270112, -0.2569500, 0.2569242
8: -5.0893068, -3.8904760, -5.0894504, -3.8904757, -0.6036826, 0.6038163
9: -3.8968308, -2.8069162, -3.8969932, -2.8069162, -0.5227560, 0.5229081

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 503
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 2770
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2720
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2786
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2788
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 2730
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 3233
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2772
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 2792
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2880
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2804
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3527

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125368, upper bound: 0.0124972
time: 10.86 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125362, upper bound: 0.0125239
time: 143.48 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.6967292, -2.2939167, -3.6971245, -2.2994561, -0.6508323, 0.6651776
1: -4.0234995, -2.1068840, -4.0243158, -2.1158934, -0.7096148, 0.7337337
2: -0.4320151, -0.2510294, -0.4317862, -0.2510171, -0.0578451, 0.0574247
3: -1.7320259, -1.2389797, -1.7299953, -1.2388724, -0.2619483, 0.2577570
4: 0.0893858, 0.2676510, 0.0898128, 0.2676657, -0.0995881, 0.0987716
5: -1.5424161, -1.0449367, -1.5401591, -1.0447791, -0.2386952, 0.2339113
6: 0.0123012, 0.2835906, 0.0139762, 0.2836375, -0.0771703, 0.0739058
7: -0.8511977, -0.4270820, -0.8505532, -0.4270559, -0.2579739, 0.2569278
8: -5.0889807, -3.8882856, -5.0891390, -3.8904755, -0.6037338, 0.6088281
9: -3.8965330, -2.8043721, -3.8967013, -2.8069165, -0.5228403, 0.5283957

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 503
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 2770
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2720
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2786
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2788
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 2730
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 3233
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2772
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 2792
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2880
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2804
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3527

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125365, upper bound: 0.0125100
time: 47.31 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125367, upper bound: 0.0125367
time: 52.04 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 105.40 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 105.40
Output dim: 4, lower bound: -0.0125368, upper bound: 0.0124972
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 105.40
Output dim: 4, lower bound: -0.0125362, upper bound: 0.0125239
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 105.40
Output dim: 4, lower bound: -0.0125365, upper bound: 0.0125100
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 105.40
Output dim: 4, lower bound: -0.0125367, upper bound: 0.0125367

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.6959991, -2.2994590, -3.6960216, -2.3006258, -0.6478206, 0.6491274
1: -4.0253434, -2.1180224, -4.0248723, -2.1182628, -0.7071176, 0.7070222
2: -0.4309359, -0.2510269, -0.4308330, -0.2514642, -0.0561181, 0.0564561
3: -1.7299951, -1.2407086, -1.7286763, -1.2409364, -0.2555262, 0.2543492
4: 0.0903727, 0.2676247, 0.0904475, 0.2673535, -0.0979030, 0.0981089
5: -1.5401552, -1.0451784, -1.5399566, -1.0450988, -0.2333206, 0.2328372
6: 0.0143412, 0.2834415, 0.0143834, 0.2833302, -0.0733274, 0.0734059
7: -0.8492832, -0.4270347, -0.8490037, -0.4272729, -0.2554005, 0.2553573
8: -5.0884910, -3.8904858, -5.0885572, -3.8909929, -0.6022478, 0.6028590
9: -3.8968282, -2.8079128, -3.8965156, -2.8080122, -0.5216433, 0.5214077

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 503
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2720
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2786
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2788
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 2730
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2772
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2792
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2880
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2804
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125098, upper bound: 0.0124970
time: 21.26 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125364, upper bound: 0.0124972
time: 5.51 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.6976013, -2.2994564, -3.6979856, -2.2994566, -0.6507422, 0.6497593
1: -4.0253515, -2.1160312, -4.0259962, -2.1160469, -0.7062154, 0.7099977
2: -0.4317367, -0.2510268, -0.4317313, -0.2510137, -0.0573737, 0.0561251
3: -1.7299957, -1.2387646, -1.7299962, -1.2386401, -0.2555160, 0.2577370
4: 0.0898472, 0.2676256, 0.0898511, 0.2676522, -0.0987381, 0.0979482
5: -1.5401621, -1.0447800, -1.5401628, -1.0446454, -0.2331834, 0.2338659
6: 0.0139978, 0.2834427, 0.0140003, 0.2835374, -0.0738756, 0.0732499
7: -0.8504743, -0.4270343, -0.8504660, -0.4270112, -0.2568635, 0.2560841
8: -5.0892949, -3.8904760, -5.0894375, -3.8904755, -0.6036732, 0.6024005
9: -3.8968313, -2.8069832, -3.8969936, -2.8069911, -0.5212431, 0.5228390

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 503
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2720
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 2786
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2788
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 2730
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2772
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2792
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2880
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2804
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3509

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125102, upper bound: 0.0125243
time: 9.19 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125356, upper bound: 0.0124966
time: 57.75 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.6951137, -2.2939186, -3.6951442, -2.3006258, -0.6479043, 0.6631773
1: -4.0234952, -2.1090112, -4.0231910, -2.1182628, -0.7073059, 0.7306159
2: -0.4311631, -0.2510295, -0.4308306, -0.2514675, -0.0565387, 0.0564665
3: -1.7320255, -1.2409346, -1.7286758, -1.2411807, -0.2596270, 0.2543631
4: 0.0899461, 0.2676501, 0.0904481, 0.2673670, -0.0987259, 0.0981330
5: -1.5424098, -1.0454084, -1.5399526, -1.0453147, -0.2380073, 0.2328802
6: 0.0126661, 0.2835895, 0.0143837, 0.2834307, -0.0766001, 0.0734994
7: -0.8499231, -0.4270822, -0.8489983, -0.4273176, -0.2564245, 0.2553612
8: -5.0881643, -3.8882952, -5.0882454, -3.8909934, -0.6022984, 0.6078708
9: -3.8965297, -2.8053684, -3.8962235, -2.8080125, -0.5217277, 0.5268962

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 503
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2720
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2786
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2788
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 2730
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2772
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2792
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2880
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2804
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3509

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125103, upper bound: 0.0125092
time: 94.63 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125361, upper bound: 0.0125092
time: 86.94 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.6967144, -2.2939162, -3.6971083, -2.2994566, -0.6508257, 0.6638083
1: -4.0234995, -2.1070209, -4.0243158, -2.1160469, -0.7064036, 0.7335923
2: -0.4319638, -0.2510294, -0.4317288, -0.2510171, -0.0577943, 0.0561356
3: -1.7320259, -1.2389908, -1.7299950, -1.2388848, -0.2596169, 0.2577513
4: 0.0894206, 0.2676511, 0.0898516, 0.2676657, -0.0995609, 0.0979723
5: -1.5424161, -1.0450103, -1.5401592, -1.0448613, -0.2378701, 0.2339092
6: 0.0123231, 0.2835904, 0.0140005, 0.2836376, -0.0771481, 0.0733434
7: -0.8511147, -0.4270820, -0.8504607, -0.4270559, -0.2578873, 0.2560877
8: -5.0889692, -3.8882859, -5.0891261, -3.8904757, -0.6037245, 0.6074125
9: -3.8965340, -2.8044391, -3.8967018, -2.8069916, -0.5213274, 0.5283260

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 503
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2720
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 2786
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2788
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 2730
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2772
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2792
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2880
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2804
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125096, upper bound: 0.0125340
time: 34.02 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125355, upper bound: 0.0125365
time: 8.03 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 48.18 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 48.18
Output dim: 4, lower bound: -0.0125098, upper bound: 0.0124970
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 48.18
Output dim: 4, lower bound: -0.0125364, upper bound: 0.0124972
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 48.18
Output dim: 4, lower bound: -0.0125102, upper bound: 0.0125243
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 48.18
Output dim: 4, lower bound: -0.0125356, upper bound: 0.0124966
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 48.18
Output dim: 4, lower bound: -0.0125103, upper bound: 0.0125092
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 48.18
Output dim: 4, lower bound: -0.0125361, upper bound: 0.0125092
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 48.18
Output dim: 4, lower bound: -0.0125096, upper bound: 0.0125340
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 48.18
Output dim: 4, lower bound: -0.0125355, upper bound: 0.0125365

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.6959867, -2.2994928, -3.6960101, -2.3006558, -0.6477840, 0.6262177
1: -4.0253458, -2.1180549, -4.0248728, -2.1182923, -0.7070668, 0.6784549
2: -0.4309268, -0.2510269, -0.4308245, -0.2514642, -0.0552075, 0.0564455
3: -1.7299854, -1.2407166, -1.7286676, -1.2409433, -0.2500373, 0.2543390
4: 0.0903820, 0.2676246, 0.0904557, 0.2673536, -0.0969869, 0.0981029
5: -1.5401444, -1.0451794, -1.5399466, -1.0450996, -0.2263603, 0.2328268
6: 0.0143635, 0.2834414, 0.0144032, 0.2833300, -0.0710219, 0.0734008
7: -0.8492795, -0.4270347, -0.8490007, -0.4272729, -0.2534465, 0.2553547
8: -5.0884905, -3.8905010, -5.0885568, -3.8910060, -0.6022173, 0.5885975
9: -3.8968287, -2.8079252, -3.8965154, -2.8080237, -0.5216246, 0.5121505

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 503
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 2770
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2720
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2786
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2788
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 2730
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 3233
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2772
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 2792
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2880
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2804
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3039

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125345, upper bound: 0.0124510
time: 46.92 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125346, upper bound: 0.0124963
time: 4.63 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.6975894, -2.2994900, -3.6979742, -2.2994864, -0.6507055, 0.6268497
1: -4.0253506, -2.1160631, -4.0259967, -2.1160755, -0.7061636, 0.6814301
2: -0.4317273, -0.2510269, -0.4317229, -0.2510137, -0.0564631, 0.0561145
3: -1.7299861, -1.2387727, -1.7299874, -1.2386475, -0.2500271, 0.2577268
4: 0.0898566, 0.2676255, 0.0898594, 0.2676521, -0.0978220, 0.0979422
5: -1.5401511, -1.0447807, -1.5401530, -1.0446463, -0.2262231, 0.2338558
6: 0.0140203, 0.2834423, 0.0140202, 0.2835371, -0.0715701, 0.0732447
7: -0.8504710, -0.4270343, -0.8504629, -0.4270112, -0.2549111, 0.2560814
8: -5.0892954, -3.8904896, -5.0894380, -3.8904884, -0.6036434, 0.5881393
9: -3.8968310, -2.8069968, -3.8969936, -2.8070018, -0.5212247, 0.5135814

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 503
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 2770
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2720
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2786
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2788
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 2730
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 3233
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2772
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 2792
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2880
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2804
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3039

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125344, upper bound: 0.0124796
time: 6.63 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125345, upper bound: 0.0125218
time: 12.12 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.6950998, -2.2939527, -3.6951332, -2.3006563, -0.6478672, 0.6402674
1: -4.0234952, -2.1090441, -4.0231924, -2.1182919, -0.7072549, 0.7020488
2: -0.4311538, -0.2510295, -0.4308223, -0.2514675, -0.0556281, 0.0564560
3: -1.7320157, -1.2409424, -1.7286671, -1.2411877, -0.2541378, 0.2543527
4: 0.0899554, 0.2676499, 0.0904562, 0.2673671, -0.0978099, 0.0981270
5: -1.5423987, -1.0454094, -1.5399427, -1.0453153, -0.2310471, 0.2328699
6: 0.0126885, 0.2835895, 0.0144035, 0.2834306, -0.0742945, 0.0734943
7: -0.8499197, -0.4270822, -0.8489950, -0.4273176, -0.2544715, 0.2553585
8: -5.0881643, -3.8883109, -5.0882459, -3.8910065, -0.6022685, 0.5936098
9: -3.8965316, -2.8053820, -3.8962226, -2.8080244, -0.5217094, 0.5176386

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 503
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 2770
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2720
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2786
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2788
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 2730
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 3233
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2772
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 2792
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2880
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2804
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3039

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125344, upper bound: 0.0124644
time: 88.04 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125348, upper bound: 0.0124646
time: 86.30 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.6967015, -2.2939503, -3.6970963, -2.2994869, -0.6507893, 0.6408989
1: -4.0235000, -2.1070533, -4.0243163, -2.1160765, -0.7063525, 0.7050240
2: -0.4319545, -0.2510294, -0.4317207, -0.2510171, -0.0568837, 0.0561250
3: -1.7320163, -1.2389983, -1.7299865, -1.2388917, -0.2541279, 0.2577409
4: 0.0894300, 0.2676510, 0.0898599, 0.2676657, -0.0986448, 0.0979664
5: -1.5424052, -1.0450113, -1.5401492, -1.0448622, -0.2309099, 0.2338990
6: 0.0123452, 0.2835906, 0.0140203, 0.2836376, -0.0748426, 0.0733383
7: -0.8511113, -0.4270820, -0.8504577, -0.4270559, -0.2559356, 0.2560852
8: -5.0889683, -3.8883004, -5.0891261, -3.8904886, -0.6036946, 0.5931511
9: -3.8965333, -2.8044519, -3.8967011, -2.8070037, -0.5213093, 0.5190688

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2186
type: B, layer: 1, pos: 503
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 2770
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2720
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2786
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2788
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 2730
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 3233
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2772
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 2792
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2880
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2804
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3039

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125343, upper bound: 0.0124915
time: 7.21 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125338, upper bound: 0.0125348
time: 10.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 23.83 seconds
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 23.83
Output dim: 4, lower bound: -0.0125345, upper bound: 0.0124510
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.83
Output dim: 4, lower bound: -0.0125346, upper bound: 0.0124963
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 23.83
Output dim: 4, lower bound: -0.0125344, upper bound: 0.0124796
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 23.83
Output dim: 4, lower bound: -0.0125345, upper bound: 0.0125218
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 23.83
Output dim: 4, lower bound: -0.0125344, upper bound: 0.0124644
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.83
Output dim: 4, lower bound: -0.0125348, upper bound: 0.0124646
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 23.83
Output dim: 4, lower bound: -0.0125343, upper bound: 0.0124915
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.83
Output dim: 4, lower bound: -0.0125338, upper bound: 0.0125348

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.6959858, -2.2995663, -3.6960077, -2.3007493, -0.6218623, 0.6261977
1: -4.0253458, -2.1181602, -4.0248728, -2.1184254, -0.6676372, 0.6784515
2: -0.4309232, -0.2510269, -0.4308200, -0.2514642, -0.0552028, 0.0555347
3: -1.7299649, -1.2407172, -1.7286415, -1.2409439, -0.2500314, 0.2478370
4: 0.0903876, 0.2676246, 0.0904629, 0.2673535, -0.0969801, 0.0964404
5: -1.5401225, -1.0451791, -1.5399188, -1.0450999, -0.2263582, 0.2250415
6: 0.0143878, 0.2834414, 0.0144341, 0.2833301, -0.0710205, 0.0671198
7: -0.8492287, -0.4270347, -0.8489359, -0.4272729, -0.2533959, 0.2539755
8: -5.0884905, -3.8905327, -5.0885568, -3.8910460, -0.5903302, 0.5885899
9: -3.8968277, -2.8079524, -3.8965158, -2.8080583, -0.5114334, 0.5121479

Time for backsubstitution: 6.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 503
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2720
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2786
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2788
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 2730
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2772
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2792
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2880
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2804
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3426

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125164, upper bound: 0.0124952
time: 85.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125345, upper bound: 0.0124950
time: 171.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.6950989, -2.2940261, -3.6951313, -2.3007493, -0.6219460, 0.6402473
1: -4.0234933, -2.1091490, -4.0231929, -2.1184254, -0.6678256, 0.7020454
2: -0.4311502, -0.2510295, -0.4308178, -0.2514675, -0.0556234, 0.0555452
3: -1.7319953, -1.2409432, -1.7286409, -1.2411886, -0.2541323, 0.2478511
4: 0.0899611, 0.2676499, 0.0904636, 0.2673671, -0.0978030, 0.0964646
5: -1.5423769, -1.0454096, -1.5399151, -1.0453155, -0.2310449, 0.2250847
6: 0.0127126, 0.2835891, 0.0144342, 0.2834306, -0.0742931, 0.0672139
7: -0.8498690, -0.4270822, -0.8489305, -0.4273176, -0.2544205, 0.2539792
8: -5.0881643, -3.8883419, -5.0882459, -3.8910463, -0.5903810, 0.5936019
9: -3.8965311, -2.8054099, -3.8962231, -2.8080590, -0.5115184, 0.5176356

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 503
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2720
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2786
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2788
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 2730
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2772
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2792
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2880
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2804
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3426

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125161, upper bound: 0.0125073
time: 106.33 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125342, upper bound: 0.0125070
time: 317.94 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.6967001, -2.2940238, -3.6970954, -2.2995801, -0.6248679, 0.6408787
1: -4.0235000, -2.1071572, -4.0243158, -2.1162086, -0.6669232, 0.7050214
2: -0.4319509, -0.2510294, -0.4317160, -0.2510171, -0.0568790, 0.0552144
3: -1.7319956, -1.2389995, -1.7299604, -1.2388928, -0.2541221, 0.2512392
4: 0.0894358, 0.2676510, 0.0898672, 0.2676657, -0.0986379, 0.0963041
5: -1.5423832, -1.0450114, -1.5401211, -1.0448622, -0.2309076, 0.2261139
6: 0.0123695, 0.2835904, 0.0140513, 0.2836375, -0.0748412, 0.0670578
7: -0.8510605, -0.4270820, -0.8503930, -0.4270559, -0.2558846, 0.2547076
8: -5.0889683, -3.8883319, -5.0891256, -3.8905289, -0.5918070, 0.5931435
9: -3.8965333, -2.8044796, -3.8967009, -2.8070383, -0.5111182, 0.5190661

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2186
type: A, layer: 1, pos: 503
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2720
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 2786
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2788
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 2730
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2772
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2792
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2880
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2804
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3426

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125158, upper bound: 0.0125340
time: 48.11 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125343, upper bound: 0.0125348
time: 81.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 135.93 seconds
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 135.93
Output dim: 4, lower bound: -0.0125164, upper bound: 0.0124952
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 135.93
Output dim: 4, lower bound: -0.0125345, upper bound: 0.0124950
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 135.93
Output dim: 4, lower bound: -0.0125161, upper bound: 0.0125073
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 135.93
Output dim: 4, lower bound: -0.0125342, upper bound: 0.0125070
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 135.93
Output dim: 4, lower bound: -0.0125158, upper bound: 0.0125340
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 135.93
Output dim: 4, lower bound: -0.0125343, upper bound: 0.0125348

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 31.10 + 1885.76 = 1916.85 seconds
