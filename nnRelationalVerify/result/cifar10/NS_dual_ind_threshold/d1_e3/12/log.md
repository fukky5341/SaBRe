## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 12)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0265445289


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3975929, 0.3975929)
1: (-5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4872523, 0.4872522)
2: (-0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0449089, 0.0449089)
3: (-0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1212449, 0.1212449)
4: (-0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816398, 0.0816398)
5: (-1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1236548, 0.1236548)
6: (-1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1357093, 0.1357093)
7: (-0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895890, 0.3895890)
8: (-5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3475804, 0.3475804)
9: (-4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4045125, 0.4045125)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.94 + 66.84 = 74.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0265680, upper bound: 0.0265689

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2843
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2833
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3555
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3297
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3344
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 622

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264943, upper bound: 0.0265744
time: 8.80 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265660, upper bound: 0.0265694
time: 29.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 38.38 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 38.38
Output dim: 7, lower bound: -0.0264943, upper bound: 0.0265744
NS_A2, status: Status.UNKNOWN, split count: 1, time: 38.38
Output dim: 7, lower bound: -0.0265660, upper bound: 0.0265694

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.2561941, -3.5884719, -4.2563844, -3.5883317, -0.3967668, 0.3968004
1: -5.9143372, -4.9860430, -5.9147758, -4.9857306, -0.4853486, 0.4854637
2: -0.6952477, -0.4676959, -0.6954178, -0.4675741, -0.0441729, 0.0442202
3: -0.9621357, -0.5944257, -0.9623265, -0.5942574, -0.1201656, 0.1202605
4: -0.2890111, -0.0123305, -0.2890153, -0.0123321, -0.0816153, 0.0816291
5: -1.0051750, -0.6741896, -1.0052334, -0.6741236, -0.1233503, 0.1233075
6: -1.0784116, -0.4421223, -1.0789361, -0.4417322, -0.1333211, 0.1335515
7: -0.5466098, -0.0515952, -0.5471674, -0.0508476, -0.3864862, 0.3863086
8: -5.4929576, -4.6514978, -5.4930167, -4.6516604, -0.3468910, 0.3476311
9: -4.9614029, -4.2109199, -4.9618607, -4.2105942, -0.4024972, 0.4026229

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2843
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2833
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3555
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3297
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3344
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2642

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264605, upper bound: 0.0265608
time: 62.30 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264724, upper bound: 0.0265596
time: 7.75 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.2564077, -3.5876975, -4.2564082, -3.5876970, -0.3975888, 0.3967885
1: -5.9149032, -4.9843149, -5.9149027, -4.9843140, -0.4872507, 0.4859571
2: -0.6954473, -0.4670206, -0.6954473, -0.4670202, -0.0449085, 0.0441930
3: -0.9623525, -0.5934684, -0.9623524, -0.5934670, -0.1212427, 0.1201259
4: -0.2890245, -0.0123321, -0.2890244, -0.0123321, -0.0816440, 0.0816357
5: -1.0052335, -0.6737123, -1.0052335, -0.6737099, -0.1236338, 0.1233483
6: -1.0789578, -0.4399688, -1.0789576, -0.4399663, -0.1357056, 0.1331088
7: -0.5497028, -0.0507913, -0.5497041, -0.0507913, -0.3894756, 0.3895838
8: -5.4932423, -4.6516600, -5.4932513, -4.6516600, -0.3484669, 0.3473676
9: -4.9619884, -4.2091150, -4.9619889, -4.2091141, -0.4045116, 0.4028416

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2843
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2833
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3555
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3297
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3344
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2642

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265444, upper bound: 0.0265631
time: 30.13 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265602, upper bound: 0.0265597
time: 172.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 208.35 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 208.35
Output dim: 7, lower bound: -0.0264605, upper bound: 0.0265608
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 208.35
Output dim: 7, lower bound: -0.0264724, upper bound: 0.0265596
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 208.35
Output dim: 7, lower bound: -0.0265444, upper bound: 0.0265631
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 208.35
Output dim: 7, lower bound: -0.0265602, upper bound: 0.0265597

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4.2556796, -3.5884719, -4.2557831, -3.5883327, -0.3959069, 0.3958259
1: -5.9134283, -4.9860430, -5.9137087, -4.9857302, -0.4835331, 0.4833868
2: -0.6952262, -0.4676959, -0.6953920, -0.4675741, -0.0441378, 0.0441800
3: -0.9621015, -0.5944283, -0.9622860, -0.5942605, -0.1201211, 0.1202091
4: -0.2890099, -0.0123717, -0.2890139, -0.0123814, -0.0815678, 0.0815887
5: -1.0051744, -0.6742460, -1.0052330, -0.6741902, -0.1232714, 0.1232379
6: -1.0784106, -0.4421832, -1.0789351, -0.4418036, -0.1332000, 0.1334483
7: -0.5466096, -0.0518582, -0.5471672, -0.0511617, -0.3861446, 0.3860201
8: -5.4915934, -4.6514988, -5.4914370, -4.6516614, -0.3443152, 0.3446347
9: -4.9604645, -4.2109213, -4.9607620, -4.2105956, -0.4002002, 0.4000890

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2843
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2833
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3555
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3297
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3344
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264451, upper bound: 0.0265531
time: 50.76 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264551, upper bound: 0.0265559
time: 7.24 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.2556572, -3.5884726, -4.2558413, -3.5861912, -0.3996435, 0.3958722
1: -5.9127073, -4.9860430, -5.9130688, -4.9815168, -0.4914785, 0.4834901
2: -0.6951663, -0.4676959, -0.6953350, -0.4675285, -0.0442829, 0.0441756
3: -0.9620210, -0.5944299, -0.9622030, -0.5941805, -0.1202358, 0.1201981
4: -0.2890093, -0.0123854, -0.2891391, -0.0123950, -0.0815664, 0.0817181
5: -1.0051742, -0.6742836, -1.0053992, -0.6742293, -0.1232782, 0.1235524
6: -1.0784098, -0.4422783, -1.0791398, -0.4419088, -0.1332004, 0.1339376
7: -0.5466095, -0.0518636, -0.5481935, -0.0511169, -0.3862540, 0.3872679
8: -5.4914064, -4.6514988, -5.4913955, -4.6456356, -0.3557977, 0.3448404
9: -4.9597192, -4.2109222, -4.9600730, -4.2068324, -0.4067562, 0.4013319

Time for backsubstitution: 6.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2843
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2833
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3555
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3297
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3344
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3539

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264596, upper bound: 0.0265565
time: 7.64 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264682, upper bound: 0.0265551
time: 55.96 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.2558937, -3.5876980, -4.2558079, -3.5876980, -0.3967290, 0.3958141
1: -5.9139929, -4.9843144, -5.9138365, -4.9843140, -0.4854355, 0.4838800
2: -0.6954258, -0.4670206, -0.6954214, -0.4670203, -0.0448736, 0.0441529
3: -0.9623183, -0.5934711, -0.9623117, -0.5934701, -0.1211983, 0.1200746
4: -0.2890230, -0.0123731, -0.2890226, -0.0123814, -0.0815966, 0.0815952
5: -1.0052333, -0.6737682, -1.0052333, -0.6737758, -0.1235550, 0.1232787
6: -1.0789566, -0.4400291, -1.0789564, -0.4400371, -0.1355846, 0.1330056
7: -0.5497027, -0.0510544, -0.5497038, -0.0511055, -0.3891345, 0.3892950
8: -5.4918795, -4.6516609, -5.4916716, -4.6516614, -0.3458910, 0.3443711
9: -4.9610505, -4.2091150, -4.9608908, -4.2091146, -0.4022145, 0.4003075

Time for backsubstitution: 6.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2843
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2833
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3555
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3297
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3344
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265308, upper bound: 0.0265558
time: 12.34 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265409, upper bound: 0.0265561
time: 66.17 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.2558708, -3.5876985, -4.2558656, -3.5855565, -0.4004655, 0.3958603
1: -5.9132719, -4.9843144, -5.9131951, -4.9800997, -0.4933816, 0.4839832
2: -0.6953658, -0.4670206, -0.6953647, -0.4669748, -0.0450186, 0.0441485
3: -0.9622377, -0.5934731, -0.9622282, -0.5933902, -0.1213130, 0.1200636
4: -0.2890221, -0.0123869, -0.2891477, -0.0123950, -0.0815952, 0.0817247
5: -1.0052330, -0.6738058, -1.0053997, -0.6738148, -0.1235619, 0.1235933
6: -1.0789561, -0.4401245, -1.0791614, -0.4401428, -0.1355850, 0.1334950
7: -0.5497028, -0.0510598, -0.5507301, -0.0510607, -0.3892437, 0.3905429
8: -5.4916916, -4.6516609, -5.4916296, -4.6456342, -0.3573750, 0.3445773
9: -4.9603043, -4.2091160, -4.9602017, -4.2053509, -0.4087708, 0.4015502

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2843
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2833
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3555
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3297
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3344
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265448, upper bound: 0.0265551
time: 9.79 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265543, upper bound: 0.0265560
time: 28.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 44.52 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 44.52
Output dim: 7, lower bound: -0.0264451, upper bound: 0.0265531
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 44.52
Output dim: 7, lower bound: -0.0264551, upper bound: 0.0265559
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 44.52
Output dim: 7, lower bound: -0.0264596, upper bound: 0.0265565
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 44.52
Output dim: 7, lower bound: -0.0264682, upper bound: 0.0265551
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 44.52
Output dim: 7, lower bound: -0.0265308, upper bound: 0.0265558
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 44.52
Output dim: 7, lower bound: -0.0265409, upper bound: 0.0265561
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 44.52
Output dim: 7, lower bound: -0.0265448, upper bound: 0.0265551
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 44.52
Output dim: 7, lower bound: -0.0265543, upper bound: 0.0265560

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.2556672, -3.5885513, -4.2557731, -3.5884013, -0.3957629, 0.3956786
1: -5.9134097, -4.9865999, -5.9136934, -4.9862146, -0.4830480, 0.4828433
2: -0.6952224, -0.4678854, -0.6953887, -0.4677391, -0.0439694, 0.0439951
3: -0.9620981, -0.5947608, -0.9622830, -0.5945501, -0.1198307, 0.1198828
4: -0.2887732, -0.0123717, -0.2888080, -0.0123814, -0.0813309, 0.0813823
5: -1.0051740, -0.6743635, -1.0052328, -0.6742942, -0.1231761, 0.1231291
6: -1.0784098, -0.4432314, -1.0789342, -0.4427171, -0.1322832, 0.1324360
7: -0.5462739, -0.0518672, -0.5468732, -0.0511699, -0.3858042, 0.3857206
8: -5.4913459, -4.6514993, -5.4912205, -4.6516619, -0.3439675, 0.3443009
9: -4.9603686, -4.2120018, -4.9606791, -4.2115364, -0.3992169, 0.3989681

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2843
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2833
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3555
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3297
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3344
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2604

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264313, upper bound: 0.0265446
time: 81.15 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264338, upper bound: 0.0265451
time: 7.67 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.2558146, -3.5885243, -4.2557812, -3.5883777, -0.3959351, 0.3958058
1: -5.9141808, -4.9860535, -5.9137001, -4.9857402, -0.4842686, 0.4829643
2: -0.6954842, -0.4677009, -0.6953901, -0.4675786, -0.0443902, 0.0440132
3: -0.9625609, -0.5944386, -0.9622850, -0.5942836, -0.1205739, 0.1199295
4: -0.2890176, -0.0120446, -0.2890130, -0.0123814, -0.0814009, 0.0819140
5: -1.0053184, -0.6742561, -1.0052329, -0.6742115, -0.1234375, 0.1231515
6: -1.0798671, -0.4421920, -1.0789350, -0.4418118, -0.1344818, 0.1325505
7: -0.5464324, -0.0515110, -0.5470024, -0.0511656, -0.3859526, 0.3862115
8: -5.4915085, -4.6512294, -5.4913583, -4.6516614, -0.3441749, 0.3446969
9: -4.9619331, -4.2109308, -4.9607553, -4.2106142, -0.4017056, 0.3993855

Time for backsubstitution: 6.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2843
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2833
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3555
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3297
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3344
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2604

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264413, upper bound: 0.0265453
time: 92.17 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264452, upper bound: 0.0265461
time: 16.17 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.2556443, -3.5885515, -4.2558303, -3.5862603, -0.3994993, 0.3957251
1: -5.9126892, -4.9865994, -5.9130535, -4.9820008, -0.4909934, 0.4829462
2: -0.6951627, -0.4678853, -0.6953321, -0.4676935, -0.0441144, 0.0439908
3: -0.9620172, -0.5947628, -0.9621999, -0.5944705, -0.1199453, 0.1198718
4: -0.2887728, -0.0123854, -0.2889327, -0.0123950, -0.0813295, 0.0815118
5: -1.0051739, -0.6744008, -1.0053989, -0.6743332, -0.1231829, 0.1234436
6: -1.0784096, -0.4433272, -1.0791396, -0.4428224, -0.1322835, 0.1329254
7: -0.5462737, -0.0518726, -0.5478994, -0.0511246, -0.3859135, 0.3869686
8: -5.4911594, -4.6514993, -5.4911795, -4.6456356, -0.3554500, 0.3445064
9: -4.9596238, -4.2120023, -4.9599900, -4.2077723, -0.4057721, 0.4002111

Time for backsubstitution: 6.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2843
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2833
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3555
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3297
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3344
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2604

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0264474, upper bound: 0.0265436
time: 6.76 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0264503, upper bound: 0.0265442
time: 37.00 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.2557917, -3.5885243, -4.2558379, -3.5862367, -0.3996714, 0.3958522
1: -5.9134607, -4.9860544, -5.9130607, -4.9815254, -0.4922141, 0.4830672
2: -0.6954242, -0.4677008, -0.6953334, -0.4675331, -0.0445352, 0.0440088
3: -0.9624798, -0.5944407, -0.9622011, -0.5942037, -0.1206885, 0.1199186
4: -0.2890168, -0.0120584, -0.2891378, -0.0123950, -0.0813994, 0.0820435
5: -1.0053184, -0.6742935, -1.0053993, -0.6742510, -0.1234442, 0.1234661
6: -1.0798664, -0.4422871, -1.0791401, -0.4419174, -0.1344821, 0.1330398
7: -0.5464321, -0.0515162, -0.5480287, -0.0511208, -0.3860621, 0.3874592
8: -5.4913211, -4.6512294, -5.4913168, -4.6456356, -0.3556574, 0.3449026
9: -4.9611883, -4.2109323, -4.9600668, -4.2068515, -0.4082615, 0.4006284

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2843
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2833
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3555
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3297
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3344
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2604

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264559, upper bound: 0.0265465
time: 31.02 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264599, upper bound: 0.0265474
time: 7.04 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.2558813, -3.5877776, -4.2557969, -3.5877671, -0.3965850, 0.3956671
1: -5.9139748, -4.9848709, -5.9138203, -4.9847984, -0.4849503, 0.4833364
2: -0.6954222, -0.4672101, -0.6954184, -0.4671852, -0.0447051, 0.0439681
3: -0.9623151, -0.5938034, -0.9623088, -0.5937600, -0.1209079, 0.1197482
4: -0.2887862, -0.0123731, -0.2888166, -0.0123814, -0.0813596, 0.0813890
5: -1.0052327, -0.6738859, -1.0052328, -0.6738796, -0.1234597, 0.1231698
6: -1.0789561, -0.4410779, -1.0789559, -0.4409502, -0.1346677, 0.1319933
7: -0.5493671, -0.0510635, -0.5494099, -0.0511131, -0.3887938, 0.3889956
8: -5.4916310, -4.6516614, -5.4914541, -4.6516619, -0.3455435, 0.3440372
9: -4.9609556, -4.2101965, -4.9608078, -4.2100554, -0.4012312, 0.3991865

Time for backsubstitution: 6.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2843
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2833
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3555
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3297
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3344
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2604

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265186, upper bound: 0.0265506
time: 6.16 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265223, upper bound: 0.0265522
time: 5.45 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.2560287, -3.5877504, -4.2558050, -3.5877435, -0.3967570, 0.3957939
1: -5.9147463, -4.9843249, -5.9138279, -4.9843225, -0.4861706, 0.4834574
2: -0.6956836, -0.4670256, -0.6954195, -0.4670247, -0.0451259, 0.0439861
3: -0.9627776, -0.5934813, -0.9623102, -0.5934932, -0.1216511, 0.1197949
4: -0.2890305, -0.0120460, -0.2890220, -0.0123814, -0.0814296, 0.0819206
5: -1.0053775, -0.6737783, -1.0052333, -0.6737974, -0.1237211, 0.1231923
6: -1.0804131, -0.4400382, -1.0789565, -0.4400454, -0.1368664, 0.1321078
7: -0.5495255, -0.0507067, -0.5495392, -0.0511091, -0.3889421, 0.3894865
8: -5.4917941, -4.6513915, -5.4915934, -4.6516609, -0.3457503, 0.3444337
9: -4.9625196, -4.2091250, -4.9608841, -4.2091331, -0.4037196, 0.3996040

Time for backsubstitution: 6.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2843
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2833
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3555
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3297
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3344
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2604

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265277, upper bound: 0.0265472
time: 97.93 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265313, upper bound: 0.0265503
time: 19.61 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.2558584, -3.5877776, -4.2558546, -3.5856256, -0.4003216, 0.3957134
1: -5.9132543, -4.9848709, -5.9131799, -4.9805846, -0.4928964, 0.4834396
2: -0.6953623, -0.4672100, -0.6953613, -0.4671397, -0.0448501, 0.0439636
3: -0.9622343, -0.5938051, -0.9622256, -0.5936800, -0.1210225, 0.1197372
4: -0.2887855, -0.0123869, -0.2889414, -0.0123950, -0.0813582, 0.0815183
5: -1.0052326, -0.6739236, -1.0053992, -0.6739193, -0.1234666, 0.1234844
6: -1.0789554, -0.4411734, -1.0791609, -0.4410562, -0.1346681, 0.1324827
7: -0.5493669, -0.0510687, -0.5504361, -0.0510687, -0.3889030, 0.3902432
8: -5.4914436, -4.6516619, -5.4914126, -4.6456347, -0.3570273, 0.3442433
9: -4.9602089, -4.2101960, -4.9601192, -4.2062912, -0.4077870, 0.4004294

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2843
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2833
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3555
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3297
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3344
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2604

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265311, upper bound: 0.0265469
time: 28.45 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265357, upper bound: 0.0265490
time: 20.53 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.2560058, -3.5877500, -4.2558622, -3.5856020, -0.4004937, 0.3958403
1: -5.9140248, -4.9843245, -5.9131870, -4.9801102, -0.4941171, 0.4835609
2: -0.6956239, -0.4670256, -0.6953627, -0.4669794, -0.0452709, 0.0439817
3: -0.9626968, -0.5934837, -0.9622269, -0.5934134, -0.1217658, 0.1197840
4: -0.2890298, -0.0120597, -0.2891471, -0.0123950, -0.0814282, 0.0820500
5: -1.0053778, -0.6738164, -1.0053996, -0.6738367, -0.1237279, 0.1235069
6: -1.0804129, -0.4401332, -1.0791616, -0.4401511, -0.1368668, 0.1325972
7: -0.5495254, -0.0507120, -0.5505654, -0.0510644, -0.3890515, 0.3907343
8: -5.4916062, -4.6513910, -5.4915509, -4.6456347, -0.3572342, 0.3446401
9: -4.9617743, -4.2091260, -4.9601955, -4.2053699, -0.4102761, 0.4008465

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2843
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2833
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3555
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3297
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2760
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3344
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2604

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0265399, upper bound: 0.0265433
time: 50.74 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265459, upper bound: 0.0265467
time: 101.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 158.38 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0264313, upper bound: 0.0265446
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0264338, upper bound: 0.0265451
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0264413, upper bound: 0.0265453
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0264452, upper bound: 0.0265461
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0264474, upper bound: 0.0265436
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0264503, upper bound: 0.0265442
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0264559, upper bound: 0.0265465
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0264599, upper bound: 0.0265474
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0265186, upper bound: 0.0265506
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0265223, upper bound: 0.0265522
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0265277, upper bound: 0.0265472
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0265313, upper bound: 0.0265503
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0265311, upper bound: 0.0265469
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0265357, upper bound: 0.0265490
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0265399, upper bound: 0.0265433
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 158.38
Output dim: 7, lower bound: -0.0265459, upper bound: 0.0265467

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.2511659, -3.5886130, -4.2506514, -3.5884709, -0.3906854, 0.3899474
1: -5.9076643, -4.9867225, -5.9071660, -4.9863558, -0.4763992, 0.4753219
2: -0.6950962, -0.4679012, -0.6952455, -0.4677572, -0.0437655, 0.0437652
3: -0.9620719, -0.5958842, -0.9622533, -0.5958169, -0.1182659, 0.1184964
4: -0.2887552, -0.0123721, -0.2887875, -0.0123819, -0.0813081, 0.0813579
5: -1.0051415, -0.6759570, -1.0051956, -0.6761049, -0.1211351, 0.1213286
6: -1.0782864, -0.4432752, -1.0787957, -0.4427665, -0.1320037, 0.1321664
7: -0.5462731, -0.0521339, -0.5468722, -0.0514707, -0.3854714, 0.3854274
8: -5.4881573, -4.6515331, -5.4875770, -4.6517005, -0.3403729, 0.3402172
9: -4.9585557, -4.2120228, -4.9586239, -4.2115593, -0.3972387, 0.3967360

Time for backsubstitution: 6.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2843
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2833
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3555
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3297
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3344
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0264159, upper bound: 0.0265438
time: 71.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0264309, upper bound: 0.0265445
time: 7.39 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.2544250, -3.5885534, -4.2544594, -3.5847096, -0.4003398, 0.3914106
1: -5.9118619, -4.9866047, -5.9119501, -4.9810209, -0.4897639, 0.4773515
2: -0.6951302, -0.4678857, -0.6952909, -0.4676746, -0.0441513, 0.0438977
3: -0.9620975, -0.5951416, -0.9631951, -0.5949779, -0.1186817, 0.1211877
4: -0.2887702, -0.0123725, -0.2888321, -0.0123824, -0.0813193, 0.0814017
5: -1.0051733, -0.6747049, -1.0067458, -0.6746781, -0.1217062, 0.1249815
6: -1.0783918, -0.4433551, -1.0789925, -0.4428568, -0.1321500, 0.1327095
7: -0.5462735, -0.0520607, -0.5469941, -0.0513840, -0.3855556, 0.3857992
8: -5.4904690, -4.6515002, -5.4902239, -4.6488552, -0.3474832, 0.3408098
9: -4.9597793, -4.2120018, -4.9600196, -4.2101259, -0.4004358, 0.3969774

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2843
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2833
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3555
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3297
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3344
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264196, upper bound: 0.0265451
time: 6.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264348, upper bound: 0.0265465
time: 9.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.2513132, -3.5885856, -4.2506590, -3.5884476, -0.3908577, 0.3900745
1: -5.9084353, -4.9861779, -5.9071727, -4.9858809, -0.4776201, 0.4754428
2: -0.6953579, -0.4677169, -0.6952468, -0.4675968, -0.0441863, 0.0437832
3: -0.9625345, -0.5955627, -0.9622548, -0.5955501, -0.1190092, 0.1185432
4: -0.2889997, -0.0120450, -0.2889929, -0.0123819, -0.0813781, 0.0818895
5: -1.0052862, -0.6758497, -1.0051959, -0.6760224, -0.1213965, 0.1213509
6: -1.0797440, -0.4422355, -1.0787966, -0.4418612, -0.1342022, 0.1322810
7: -0.5464313, -0.0517770, -0.5470015, -0.0514668, -0.3856200, 0.3859185
8: -5.4883194, -4.6512632, -5.4877143, -4.6517000, -0.3405802, 0.3406132
9: -4.9601197, -4.2109528, -4.9587002, -4.2106380, -0.3997272, 0.3971531

Time for backsubstitution: 6.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2843
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2833
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3555
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3297
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3344
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264239, upper bound: 0.0265455
time: 12.94 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264408, upper bound: 0.0265452
time: 17.13 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.2545729, -3.5885262, -4.2544670, -3.5846858, -0.4005119, 0.3915378
1: -5.9126334, -4.9860592, -5.9119573, -4.9805460, -0.4909849, 0.4774724
2: -0.6953919, -0.4677013, -0.6952924, -0.4675143, -0.0445721, 0.0439158
3: -0.9625599, -0.5948198, -0.9631964, -0.5947113, -0.1194250, 0.1212344
4: -0.2890145, -0.0120454, -0.2890372, -0.0123824, -0.0813892, 0.0819334
5: -1.0053177, -0.6745974, -1.0067461, -0.6745957, -0.1219674, 0.1250039
6: -1.0798495, -0.4423154, -1.0789928, -0.4419518, -0.1343485, 0.1328240
7: -0.5464317, -0.0517040, -0.5471234, -0.0513796, -0.3857042, 0.3862901
8: -5.4906306, -4.6512308, -5.4903617, -4.6488547, -0.3476906, 0.3412058
9: -4.9613457, -4.2109327, -4.9600968, -4.2092042, -0.4029245, 0.3973949

Time for backsubstitution: 6.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2843
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2833
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3555
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3297
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3344
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3539

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264294, upper bound: 0.0265469
time: 91.09 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264454, upper bound: 0.0265449
time: 135.21 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.2512903, -3.5885859, -4.2507172, -3.5863061, -0.3945937, 0.3901208
1: -5.9077153, -4.9861779, -5.9065332, -4.9816675, -0.4855654, 0.4755453
2: -0.6952981, -0.4677168, -0.6951902, -0.4675513, -0.0443314, 0.0437789
3: -0.9624536, -0.5955644, -0.9621716, -0.5954695, -0.1191238, 0.1185322
4: -0.2889989, -0.0120588, -0.2891175, -0.0123953, -0.0813767, 0.0820190
5: -1.0052860, -0.6758871, -1.0053622, -0.6760619, -0.1214033, 0.1216656
6: -1.0797433, -0.4423310, -1.0790014, -0.4419674, -0.1342027, 0.1327703
7: -0.5464314, -0.0517825, -0.5480278, -0.0514216, -0.3857297, 0.3871661
8: -5.4881325, -4.6512632, -5.4876733, -4.6456733, -0.3520625, 0.3408190
9: -4.9593754, -4.2109528, -4.9580112, -4.2068739, -0.4062830, 0.3983964

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2843
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2833
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3555
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3297
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3344
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3539

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264393, upper bound: 0.0265483
time: 7.27 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0264550, upper bound: 0.0265443
time: 55.72 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.2545500, -3.5885262, -4.2545247, -3.5825448, -0.4042484, 0.3915840
1: -5.9119139, -4.9860587, -5.9113173, -4.9763327, -0.4989300, 0.4775757
2: -0.6953321, -0.4677012, -0.6952355, -0.4674687, -0.0447172, 0.0439114
3: -0.9624789, -0.5948218, -0.9631133, -0.5946310, -0.1195396, 0.1212236
4: -0.2890138, -0.0120592, -0.2891627, -0.0123959, -0.0813878, 0.0820628
5: -1.0053178, -0.6746352, -1.0069126, -0.6746351, -0.1219743, 0.1253186
6: -1.0798488, -0.4424106, -1.0791981, -0.4420574, -0.1343489, 0.1333134
7: -0.5464316, -0.0517092, -0.5481498, -0.0513346, -0.3858138, 0.3875381
8: -5.4904442, -4.6512308, -5.4903216, -4.6428280, -0.3591728, 0.3414117
9: -4.9605999, -4.2109327, -4.9594069, -4.2054410, -0.4094801, 0.3986379

Time for backsubstitution: 6.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2843
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2833
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3555
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3297
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2760
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3344
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3539

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264429, upper bound: 0.0265449
time: 229.18 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264593, upper bound: 0.0265451
time: 131.05 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 74.78 + 2056.99 = 2131.77 seconds
