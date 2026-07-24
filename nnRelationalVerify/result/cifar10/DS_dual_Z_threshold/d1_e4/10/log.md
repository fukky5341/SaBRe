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
execution time: IAR + RelationalAnalysis = 7.80 + 23.53 = 31.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0125471, upper bound: 0.0125478

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3088

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125460, upper bound: 0.0125481
time: 7.29 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125474, upper bound: 0.0125467
time: 119.95 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 127.31 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 127.31
Output dim: 4, lower bound: -0.0125460, upper bound: 0.0125481
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 127.31
Output dim: 4, lower bound: -0.0125474, upper bound: 0.0125467

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6555204, 0.6555212
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7175524, 0.7175522
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0575601, 0.0575599
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2591877, 0.2591863
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0990128, 0.0990128
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2356459, 0.2356450
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0748268, 0.0748268
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2572220, 0.2572219
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6054889, 0.6054888
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5246340, 0.5246339

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2665

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125442, upper bound: 0.0125421
time: 189.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125409, upper bound: 0.0125461
time: 37.24 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6555212, 0.6555204
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7175522, 0.7175524
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0575599, 0.0575601
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2591863, 0.2591877
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0990128, 0.0990128
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2356450, 0.2356459
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0748268, 0.0748268
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2572220, 0.2572219
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6054888, 0.6054888
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5246340, 0.5246339

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2665

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125454, upper bound: 0.0125418
time: 9.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125421, upper bound: 0.0125442
time: 129.51 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 145.18 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 145.18
Output dim: 4, lower bound: -0.0125442, upper bound: 0.0125421
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 145.18
Output dim: 4, lower bound: -0.0125409, upper bound: 0.0125461
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 145.18
Output dim: 4, lower bound: -0.0125454, upper bound: 0.0125418
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 145.18
Output dim: 4, lower bound: -0.0125421, upper bound: 0.0125442

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6537762, 0.6537971
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7106528, 0.7107553
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0574666, 0.0574670
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2588936, 0.2588973
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0988951, 0.0988939
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2355800, 0.2355795
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747663, 0.0747653
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2571504, 0.2571509
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6000830, 0.6001724
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5182097, 0.5183084

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2396

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125230, upper bound: 0.0125397
time: 10.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125412, upper bound: 0.0125216
time: 8.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6537966, 0.6537771
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7107553, 0.7106526
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0574671, 0.0574665
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2588986, 0.2588921
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0988939, 0.0988951
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2355804, 0.2355790
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747653, 0.0747663
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2571509, 0.2571504
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6001725, 0.6000828
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5183084, 0.5182097

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2396

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125197, upper bound: 0.0125399
time: 112.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125382, upper bound: 0.0125255
time: 68.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6537770, 0.6537967
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7106526, 0.7107553
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0574665, 0.0574671
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2588922, 0.2588986
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0988951, 0.0988939
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2355790, 0.2355804
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747663, 0.0747653
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2571504, 0.2571509
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6000829, 0.6001724
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5182096, 0.5183084

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2396

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125242, upper bound: 0.0125393
time: 9.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125424, upper bound: 0.0125206
time: 114.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6537973, 0.6537762
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7107553, 0.7106528
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0574670, 0.0574666
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2588972, 0.2588936
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0988939, 0.0988951
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2355795, 0.2355800
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0747653, 0.0747663
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2571509, 0.2571504
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.6001725, 0.6000828
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5183084, 0.5182097

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2396

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125209, upper bound: 0.0125422
time: 8.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125393, upper bound: 0.0125242
time: 89.12 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 103.40 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 103.40
Output dim: 4, lower bound: -0.0125230, upper bound: 0.0125397
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 103.40
Output dim: 4, lower bound: -0.0125412, upper bound: 0.0125216
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 103.40
Output dim: 4, lower bound: -0.0125197, upper bound: 0.0125399
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 103.40
Output dim: 4, lower bound: -0.0125382, upper bound: 0.0125255
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 103.40
Output dim: 4, lower bound: -0.0125242, upper bound: 0.0125393
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 103.40
Output dim: 4, lower bound: -0.0125424, upper bound: 0.0125206
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 103.40
Output dim: 4, lower bound: -0.0125209, upper bound: 0.0125422
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 103.40
Output dim: 4, lower bound: -0.0125393, upper bound: 0.0125242

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6493582, 0.6494210
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7058747, 0.7057140
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0572361, 0.0572467
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580515, 0.2580347
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0987475, 0.0987528
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2344539, 0.2345002
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0746508, 0.0746529
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2566210, 0.2566487
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.5960063, 0.5958501
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5151891, 0.5151055

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2395

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125077, upper bound: 0.0125356
time: 140.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125174, upper bound: 0.0125132
time: 71.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6493999, 0.6493793
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7056115, 0.7059772
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0572463, 0.0572364
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580310, 0.2580552
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0987539, 0.0987464
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2345008, 0.2344535
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0746538, 0.0746499
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2566482, 0.2566215
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.5957607, 0.5960957
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5150067, 0.5152876

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2395

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125156, upper bound: 0.0125162
time: 72.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125383, upper bound: 0.0125075
time: 8.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6493785, 0.6494005
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7059772, 0.7056112
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0572365, 0.0572462
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580565, 0.2580296
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0987463, 0.0987540
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2344544, 0.2344999
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0746498, 0.0746539
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2566215, 0.2566482
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.5960958, 0.5957607
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5152878, 0.5150068

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2395

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125047, upper bound: 0.0125402
time: 75.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125140, upper bound: 0.0125170
time: 268.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6494201, 0.6493590
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7057140, 0.7058744
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0572468, 0.0572360
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580361, 0.2580501
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0987528, 0.0987475
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2345012, 0.2344530
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0746528, 0.0746509
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2566487, 0.2566211
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.5958502, 0.5960063
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5151054, 0.5151889

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2395

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125123, upper bound: 0.0125189
time: 15.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125350, upper bound: 0.0125086
time: 28.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6493590, 0.6494200
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7058744, 0.7057140
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0572360, 0.0572468
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580501, 0.2580361
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0987475, 0.0987528
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2344530, 0.2345013
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0746509, 0.0746528
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2566210, 0.2566487
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.5960062, 0.5958503
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5151891, 0.5151055

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2395

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125089, upper bound: 0.0125342
time: 21.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125186, upper bound: 0.0125129
time: 58.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6494006, 0.6493785
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7056112, 0.7059772
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0572462, 0.0572366
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580296, 0.2580565
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0987540, 0.0987463
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2344998, 0.2344545
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0746539, 0.0746498
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2566482, 0.2566215
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.5957606, 0.5960959
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5150065, 0.5152876

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2730
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2720
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2772
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2792
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2804
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2395

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125168, upper bound: 0.0125145
time: 8.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125395, upper bound: 0.0125053
time: 7.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7011218, -2.2994559, -3.7011218, -2.2994559, -0.6493793, 0.6493998
1: -4.0311794, -2.1158938, -4.0311794, -2.1158938, -0.7059772, 0.7056115
2: -0.4317924, -0.2508998, -0.4317924, -0.2508998, -0.0572364, 0.0572463
3: -1.7299969, -1.2375873, -1.7299969, -1.2375873, -0.2580551, 0.2580310
4: 0.0898111, 0.2678648, 0.0898111, 0.2678648, -0.0987464, 0.0987539
5: -1.5401679, -1.0433666, -1.5401679, -1.0433666, -0.2344534, 0.2345008
6: 0.0139757, 0.2843854, 0.0139757, 0.2843854, -0.0746499, 0.0746538
7: -0.8505678, -0.4268264, -0.8505678, -0.4268264, -0.2566215, 0.2566482
8: -5.0906124, -3.8904743, -5.0906124, -3.8904743, -0.5960958, 0.5957607
9: -3.8983092, -2.8069162, -3.8983092, -2.8069162, -0.5152878, 0.5150068

Time for backsubstitution: 5.94 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 31.33 + 1769.08 = 1800.41 seconds
