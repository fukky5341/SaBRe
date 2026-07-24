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
execution time: IAR + RelationalAnalysis = 7.91 + 23.45 = 31.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0125471, upper bound: 0.0125478

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2149

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125422, upper bound: 0.0125476
time: 109.50 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125469, upper bound: 0.0125427
time: 9.27 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 118.79 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 118.79
Output dim: 4, lower bound: -0.0125422, upper bound: 0.0125476
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 118.79
Output dim: 4, lower bound: -0.0125469, upper bound: 0.0125427

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6522007, 0.6519935
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7124387, 0.7121296
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0573780, 0.0573849
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2579652, 0.2580325
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0989170, 0.0989214
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2341400, 0.2342245
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747858, 0.0747861
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2568030, 0.2568250
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6029296, 0.6027651
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5217248, 0.5215437

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2792

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2736

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125363, upper bound: 0.0125475
time: 8.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125417, upper bound: 0.0125370
time: 137.68 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6519934, 0.6522009
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7121295, 0.7124386
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0573849, 0.0573780
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580325, 0.2579652
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0989214, 0.0989170
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2342244, 0.2341400
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747861, 0.0747858
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2568249, 0.2568030
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6027651, 0.6029296
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5215439, 0.5217249

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2864

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125468, upper bound: 0.0125419
time: 78.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125468, upper bound: 0.0125425
time: 25.43 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 109.60 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 109.60
Output dim: 4, lower bound: -0.0125363, upper bound: 0.0125475
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 109.60
Output dim: 4, lower bound: -0.0125417, upper bound: 0.0125370
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 109.60
Output dim: 4, lower bound: -0.0125468, upper bound: 0.0125419
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 109.60
Output dim: 4, lower bound: -0.0125468, upper bound: 0.0125425

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6521912, 0.6519747
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7123432, 0.7119875
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0573605, 0.0573701
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2579590, 0.2580230
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0989033, 0.0989141
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2341355, 0.2342207
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747507, 0.0747616
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2567999, 0.2568196
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6029115, 0.6027310
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5216535, 0.5214257

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3070

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125115, upper bound: 0.0125455
time: 91.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125359, upper bound: 0.0125215
time: 28.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6521820, 0.6519837
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7122965, 0.7120342
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0573632, 0.0573675
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2579557, 0.2580263
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0989097, 0.0989076
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2341362, 0.2342200
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747613, 0.0747510
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2567978, 0.2568218
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6028953, 0.6027472
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5216068, 0.5214725

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2819

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2151

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125176, upper bound: 0.0125406
time: 129.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125410, upper bound: 0.0125176
time: 8.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6519934, 0.6522009
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7121295, 0.7124386
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0573849, 0.0573780
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580325, 0.2579652
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0989214, 0.0989170
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2342244, 0.2341400
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747861, 0.0747858
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2568249, 0.2568030
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6027651, 0.6029296
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5215439, 0.5217249

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2151

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 522

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125435, upper bound: 0.0125430
time: 101.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125472, upper bound: 0.0125381
time: 156.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6519934, 0.6522009
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7121295, 0.7124386
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0573849, 0.0573780
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580325, 0.2579652
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0989214, 0.0989170
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2342244, 0.2341400
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747861, 0.0747858
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2568249, 0.2568030
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6027651, 0.6029296
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5215439, 0.5217249

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 600

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3508

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125369, upper bound: 0.0125427
time: 103.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125474, upper bound: 0.0125325
time: 16.90 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 126.89 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 126.89
Output dim: 4, lower bound: -0.0125115, upper bound: 0.0125455
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 126.89
Output dim: 4, lower bound: -0.0125359, upper bound: 0.0125215
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 126.89
Output dim: 4, lower bound: -0.0125176, upper bound: 0.0125406
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 126.89
Output dim: 4, lower bound: -0.0125410, upper bound: 0.0125176
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 126.89
Output dim: 4, lower bound: -0.0125435, upper bound: 0.0125430
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 126.89
Output dim: 4, lower bound: -0.0125472, upper bound: 0.0125381
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 126.89
Output dim: 4, lower bound: -0.0125369, upper bound: 0.0125427
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 126.89
Output dim: 4, lower bound: -0.0125474, upper bound: 0.0125325

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6302745, 0.6295919
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.6881289, 0.6871743
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0562694, 0.0563059
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2534612, 0.2536258
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0982175, 0.0982457
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2283497, 0.2285802
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0740960, 0.0741232
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2543171, 0.2543966
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.5831267, 0.5825236
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5076666, 0.5070770

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2780

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2957

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125118, upper bound: 0.0125409
time: 102.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125117, upper bound: 0.0125466
time: 6.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6298087, 0.6300578
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.6875303, 0.6877730
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0562963, 0.0562790
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2535619, 0.2535252
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0982350, 0.0982283
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2284951, 0.2284349
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0741123, 0.0741069
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2543768, 0.2543368
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.5827042, 0.5829461
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5073048, 0.5074388

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 807

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125211, upper bound: 0.0125215
time: 5.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125356, upper bound: 0.0125058
time: 5.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6407799, 0.6402799
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.6960778, 0.6954048
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0571176, 0.0571291
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2544271, 0.2545879
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0985395, 0.0985491
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2298144, 0.2300118
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0733048, 0.0733309
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2561219, 0.2561638
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.5963364, 0.5960466
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5159740, 0.5157067

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2786

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2803

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125152, upper bound: 0.0125411
time: 52.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125178, upper bound: 0.0125375
time: 148.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6404779, 0.6405817
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.6956670, 0.6958153
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0571248, 0.0571219
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2545173, 0.2544977
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0985511, 0.0985374
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2299281, 0.2298982
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0733412, 0.0732946
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2561398, 0.2561458
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.5961947, 0.5961882
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5158409, 0.5158397

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3540

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125342, upper bound: 0.0125171
time: 8.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125411, upper bound: 0.0125101
time: 6.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6519822, 0.6521885
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7120708, 0.7123880
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0573803, 0.0573725
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580238, 0.2579532
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0989127, 0.0989095
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2342109, 0.2341244
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747470, 0.0747528
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2568099, 0.2567853
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6026510, 0.6028341
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5214528, 0.5216495

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 521

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2267

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125435, upper bound: 0.0125416
time: 70.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125423, upper bound: 0.0125428
time: 6.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6519812, 0.6521895
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7120789, 0.7123799
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0573794, 0.0573734
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580205, 0.2579565
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0989138, 0.0989083
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2342088, 0.2341265
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747531, 0.0747467
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2568071, 0.2567881
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6026697, 0.6028155
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5214684, 0.5216337

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2786

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125402, upper bound: 0.0125381
time: 8.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125462, upper bound: 0.0125324
time: 129.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6514498, 0.6517004
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7119395, 0.7122402
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0573047, 0.0573014
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2579682, 0.2578864
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0988151, 0.0988167
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2340842, 0.2340077
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0746216, 0.0746300
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2564437, 0.2564478
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6023463, 0.6025120
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5211188, 0.5213025

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 836

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125350, upper bound: 0.0125417
time: 10.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125352, upper bound: 0.0125413
time: 223.64 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 240.54 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125118, upper bound: 0.0125409
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125117, upper bound: 0.0125466
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125211, upper bound: 0.0125215
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125356, upper bound: 0.0125058
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125152, upper bound: 0.0125411
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125178, upper bound: 0.0125375
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125342, upper bound: 0.0125171
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125411, upper bound: 0.0125101
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125435, upper bound: 0.0125416
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125423, upper bound: 0.0125428
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125402, upper bound: 0.0125381
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125462, upper bound: 0.0125324
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125350, upper bound: 0.0125417
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 240.54
Output dim: 4, lower bound: -0.0125352, upper bound: 0.0125413
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 240.54
Output dim: 4, lower bound: -0.0125474, upper bound: 0.0125325

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 31.35 + 1870.87 = 1902.22 seconds
