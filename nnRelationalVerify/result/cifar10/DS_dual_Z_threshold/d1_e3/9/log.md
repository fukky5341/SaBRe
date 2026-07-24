## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 9)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0059452488000000005


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4746015, 0.4746014)
1: (-4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6756127, 0.6756127)
2: (-0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371262, 0.2371262)
3: (-1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1957060, 0.1957060)
4: (0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017926, 0.2017926)
5: (-1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2293791, 0.2293791)
6: (0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348633, 0.0348633)
7: (-2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1252365, 0.1252365)
8: (-4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5847010, 0.5847009)
9: (-4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5155784, 0.5155784)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.79 + 45.73 = 53.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0059458, upper bound: 0.0059506

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3123

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059394, upper bound: 0.0059475
time: 29.22 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059461, upper bound: 0.0059514
time: 5.10 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 34.39 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 34.39
Output dim: 6, lower bound: -0.0059394, upper bound: 0.0059475
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 34.39
Output dim: 6, lower bound: -0.0059461, upper bound: 0.0059514

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4745922, 0.4745922
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6756278, 0.6756279
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371325, 0.2371325
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1957074, 0.1957074
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017950, 0.2017950
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2293788, 0.2293788
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348681, 0.0348681
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1252241, 0.1252242
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5847004, 0.5847005
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5155962, 0.5155964

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3109

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059325, upper bound: 0.0059406
time: 10.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059377, upper bound: 0.0059389
time: 33.19 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4745922, 0.4745922
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6756279, 0.6756279
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371325, 0.2371325
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1957074, 0.1957074
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017950, 0.2017949
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2293788, 0.2293788
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348681, 0.0348681
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1252242, 0.1252241
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5847004, 0.5847005
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5155964, 0.5155963

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3109

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059309, upper bound: 0.0059394
time: 9.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059381, upper bound: 0.0059380
time: 35.33 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 50.61 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 50.61
Output dim: 6, lower bound: -0.0059325, upper bound: 0.0059406
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 50.61
Output dim: 6, lower bound: -0.0059377, upper bound: 0.0059389
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 50.61
Output dim: 6, lower bound: -0.0059309, upper bound: 0.0059394
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 50.61
Output dim: 6, lower bound: -0.0059381, upper bound: 0.0059380

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 53.51 + 134.71 = 188.22 seconds
