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
execution time: IAR + RelationalAnalysis = 8.32 + 46.64 = 54.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0059458, upper bound: 0.0059506

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2262

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2845

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059386, upper bound: 0.0059488
time: 10.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059450, upper bound: 0.0059417
time: 9.03 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 19.32 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 19.32
Output dim: 6, lower bound: -0.0059386, upper bound: 0.0059488
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 19.32
Output dim: 6, lower bound: -0.0059450, upper bound: 0.0059417

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4742177, 0.4741885
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6751841, 0.6751496
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371236, 0.2371207
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1956390, 0.1956436
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017919, 0.2017911
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2293105, 0.2293153
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348549, 0.0348577
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1252275, 0.1252234
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5843627, 0.5843349
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5153291, 0.5152986

Time for backsubstitution: 6.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2787

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2100

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059395, upper bound: 0.0059487
time: 9.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059379, upper bound: 0.0059482
time: 9.12 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 24.56 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 24.56
Output dim: 6, lower bound: -0.0059395, upper bound: 0.0059487
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 24.56
Output dim: 6, lower bound: -0.0059379, upper bound: 0.0059482

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4741881, 0.4741454
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6751758, 0.6751364
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371217, 0.2371184
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1956372, 0.1956420
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017911, 0.2017908
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2293093, 0.2293141
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348540, 0.0348568
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1252244, 0.1252180
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5843060, 0.5842510
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5152913, 0.5152753

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 3232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2409

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059366, upper bound: 0.0059466
time: 8.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059382, upper bound: 0.0059490
time: 6.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4741746, 0.4741588
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6751709, 0.6751412
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371213, 0.2371188
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1956375, 0.1956417
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017916, 0.2017903
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2293094, 0.2293141
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348540, 0.0348568
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1252222, 0.1252203
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5842789, 0.5842781
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5153058, 0.5152609

Time for backsubstitution: 6.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2643

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2621

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059359, upper bound: 0.0059480
time: 63.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059406, upper bound: 0.0059452
time: 7.11 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 76.76 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 76.76
Output dim: 6, lower bound: -0.0059366, upper bound: 0.0059466
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 76.76
Output dim: 6, lower bound: -0.0059382, upper bound: 0.0059490
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 76.76
Output dim: 6, lower bound: -0.0059359, upper bound: 0.0059480
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 76.76
Output dim: 6, lower bound: -0.0059406, upper bound: 0.0059452

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4732581, 0.4731469
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6737658, 0.6735841
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371061, 0.2371023
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1955303, 0.1955157
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017767, 0.2017770
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2291414, 0.2291396
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348463, 0.0348493
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1251516, 0.1251628
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5831673, 0.5829967
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5144842, 0.5143495

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2702

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 815

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059359, upper bound: 0.0059422
time: 8.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059356, upper bound: 0.0059435
time: 9.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4731895, 0.4732149
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6736234, 0.6737263
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371057, 0.2371028
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1955108, 0.1955352
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017773, 0.2017764
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2291347, 0.2291462
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348464, 0.0348491
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1251692, 0.1251452
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5830517, 0.5831124
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5143657, 0.5144718

Time for backsubstitution: 6.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 3070

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3447

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059357, upper bound: 0.0059295
time: 7.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059174, upper bound: 0.0059451
time: 8.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4680479, 0.4682163
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6684497, 0.6686559
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371402, 0.2371365
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1942226, 0.1941869
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017778, 0.2017763
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2278097, 0.2277707
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348222, 0.0348261
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1247385, 0.1247255
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5777617, 0.5779635
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5106256, 0.5107151

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2786

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2892

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059349, upper bound: 0.0059494
time: 44.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059333, upper bound: 0.0059445
time: 32.16 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 82.82 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 82.82
Output dim: 6, lower bound: -0.0059359, upper bound: 0.0059422
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 82.82
Output dim: 6, lower bound: -0.0059356, upper bound: 0.0059435
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 82.82
Output dim: 6, lower bound: -0.0059357, upper bound: 0.0059295
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 82.82
Output dim: 6, lower bound: -0.0059174, upper bound: 0.0059451
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 82.82
Output dim: 6, lower bound: -0.0059349, upper bound: 0.0059494
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 82.82
Output dim: 6, lower bound: -0.0059333, upper bound: 0.0059445

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4680160, 0.4681828
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6684310, 0.6686363
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371397, 0.2371359
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1941780, 0.1941328
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017723, 0.2017708
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2277755, 0.2277296
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348214, 0.0348253
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1247153, 0.1247024
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5777331, 0.5779324
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5106087, 0.5106968

Time for backsubstitution: 6.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2866

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2461

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059371, upper bound: 0.0059480
time: 82.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059371, upper bound: 0.0059496
time: 9.86 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 98.93 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 98.93
Output dim: 6, lower bound: -0.0059371, upper bound: 0.0059480
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 98.93
Output dim: 6, lower bound: -0.0059371, upper bound: 0.0059496

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4680160, 0.4681828
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6684310, 0.6686363
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371397, 0.2371359
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1941780, 0.1941328
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017723, 0.2017708
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2277755, 0.2277296
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348214, 0.0348253
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1247153, 0.1247024
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5777331, 0.5779324
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5106087, 0.5106968

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 3134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 339

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059190, upper bound: 0.0059197
time: 27.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059081, upper bound: 0.0059360
time: 89.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4680160, 0.4681828
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6684310, 0.6686363
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371397, 0.2371359
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1941780, 0.1941328
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017723, 0.2017708
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2277755, 0.2277296
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0348214, 0.0348253
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1247153, 0.1247024
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5777331, 0.5779324
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5106087, 0.5106968

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2785

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2782

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059259, upper bound: 0.0059453
time: 26.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059328, upper bound: 0.0059365
time: 61.76 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 94.53 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 94.53
Output dim: 6, lower bound: -0.0059190, upper bound: 0.0059197
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 94.53
Output dim: 6, lower bound: -0.0059081, upper bound: 0.0059360
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 94.53
Output dim: 6, lower bound: -0.0059259, upper bound: 0.0059453
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 94.53
Output dim: 6, lower bound: -0.0059328, upper bound: 0.0059365

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4664922, 0.4666071
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6651855, 0.6653285
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371413, 0.2371366
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1939539, 0.1939008
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017360, 0.2017345
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2276641, 0.2276117
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0347112, 0.0347182
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1244031, 0.1243842
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5771071, 0.5773144
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5092468, 0.5093105

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2370

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2838

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0059267, upper bound: 0.0059467
time: 49.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059293, upper bound: 0.0059380
time: 111.72 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 167.79 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 167.79
Output dim: 6, lower bound: -0.0059267, upper bound: 0.0059467
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 167.79
Output dim: 6, lower bound: -0.0059293, upper bound: 0.0059380

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6438184, -2.8250866, -3.6438184, -2.8250866, -0.4660744, 0.4661837
1: -4.7243891, -3.5866122, -4.7243891, -3.5866122, -0.6642747, 0.6643958
2: -0.5330166, -0.0883134, -0.5330166, -0.0883134, -0.2371571, 0.2371521
3: -1.4352405, -1.1260650, -1.4352405, -1.1260650, -0.1939798, 0.1939256
4: 0.2493559, 0.5066521, 0.2493559, 0.5066521, -0.2017138, 0.2017127
5: -1.6566334, -1.2748274, -1.6566334, -1.2748274, -0.2276838, 0.2276303
6: 0.4361449, 0.5592507, 0.4361449, 0.5592507, -0.0347084, 0.0347154
7: -2.5218201, -2.1213250, -2.5218201, -2.1213250, -0.1244060, 0.1243867
8: -4.7069869, -3.7994916, -4.7069869, -3.7994916, -0.5768259, 0.5770282
9: -4.7287807, -3.8938146, -4.7287807, -3.8938146, -0.5088942, 0.5089505

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3212
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2707
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2786
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 559
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2719
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2787
type: DSZ, layer: 1, pos: 2766
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3378
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2728
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 336
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2788
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2822
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 3154
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 3512
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2650

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3212

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059228, upper bound: 0.0059448
time: 80.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059244, upper bound: 0.0059314
time: 64.61 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 151.26 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 151.26
Output dim: 6, lower bound: -0.0059228, upper bound: 0.0059448
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 151.26
Output dim: 6, lower bound: -0.0059244, upper bound: 0.0059314

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 54.96 + 907.11 = 962.07 seconds
