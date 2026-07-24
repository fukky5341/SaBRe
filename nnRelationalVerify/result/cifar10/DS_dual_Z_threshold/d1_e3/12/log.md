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
execution time: IAR + RelationalAnalysis = 7.84 + 66.50 = 74.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0265680, upper bound: 0.0265689

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3295

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265549, upper bound: 0.0265701
time: 6.69 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265675, upper bound: 0.0265552
time: 44.40 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 51.16 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 51.16
Output dim: 7, lower bound: -0.0265549, upper bound: 0.0265701
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 51.16
Output dim: 7, lower bound: -0.0265675, upper bound: 0.0265552

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3967124, 0.3967016
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4861324, 0.4861168
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0448894, 0.0448886
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1211284, 0.1211286
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816428, 0.0816428
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1235463, 0.1235464
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1358098, 0.1358082
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3896266, 0.3896281
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3465115, 0.3465033
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4031790, 0.4031768

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3280

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265505, upper bound: 0.0265712
time: 9.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265538, upper bound: 0.0265605
time: 20.04 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3967017, 0.3967124
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4861168, 0.4861324
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0448886, 0.0448894
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1211286, 0.1211284
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816428, 0.0816428
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1235464, 0.1235463
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1358082, 0.1358098
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3896281, 0.3896265
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3465033, 0.3465115
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4031768, 0.4031790

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3280

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265565, upper bound: 0.0265541
time: 50.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265681, upper bound: 0.0265522
time: 97.03 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 153.87 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 153.87
Output dim: 7, lower bound: -0.0265505, upper bound: 0.0265712
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 153.87
Output dim: 7, lower bound: -0.0265538, upper bound: 0.0265605
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 153.87
Output dim: 7, lower bound: -0.0265565, upper bound: 0.0265541
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 153.87
Output dim: 7, lower bound: -0.0265681, upper bound: 0.0265522

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3961771, 0.3961531
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4853809, 0.4853455
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0448747, 0.0448730
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210478, 0.1210496
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816427, 0.0816427
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1234558, 0.1234610
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1357836, 0.1357759
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3896206, 0.3896234
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3455654, 0.3455483
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4023483, 0.4023470

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2572

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265463, upper bound: 0.0265614
time: 27.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265397, upper bound: 0.0265672
time: 9.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3961537, 0.3961664
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4853461, 0.4853652
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0448735, 0.0448739
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210494, 0.1210467
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816427, 0.0816427
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1234609, 0.1234545
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1357775, 0.1357802
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3896219, 0.3896213
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3455444, 0.3455571
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4023398, 0.4023461

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2572

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265497, upper bound: 0.0265475
time: 94.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265426, upper bound: 0.0265539
time: 7.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3961664, 0.3961537
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4853653, 0.4853460
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0448739, 0.0448735
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210467, 0.1210494
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816427, 0.0816427
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1234545, 0.1234609
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1357803, 0.1357774
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3896213, 0.3896219
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3455571, 0.3455445
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4023460, 0.4023396

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2572

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265526, upper bound: 0.0265474
time: 6.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265463, upper bound: 0.0265517
time: 17.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3961532, 0.3961772
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4853455, 0.4853808
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0448730, 0.0448747
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210497, 0.1210478
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816427, 0.0816427
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1234610, 0.1234558
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1357759, 0.1357836
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3896233, 0.3896205
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3455483, 0.3455654
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4023470, 0.4023482

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2572

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265637, upper bound: 0.0265394
time: 21.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265567, upper bound: 0.0265498
time: 22.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 50.56 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 50.56
Output dim: 7, lower bound: -0.0265463, upper bound: 0.0265614
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 50.56
Output dim: 7, lower bound: -0.0265397, upper bound: 0.0265672
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 50.56
Output dim: 7, lower bound: -0.0265497, upper bound: 0.0265475
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 50.56
Output dim: 7, lower bound: -0.0265426, upper bound: 0.0265539
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 50.56
Output dim: 7, lower bound: -0.0265526, upper bound: 0.0265474
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 50.56
Output dim: 7, lower bound: -0.0265463, upper bound: 0.0265517
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 50.56
Output dim: 7, lower bound: -0.0265637, upper bound: 0.0265394
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 50.56
Output dim: 7, lower bound: -0.0265567, upper bound: 0.0265498

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3917005, 0.3918016
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4772637, 0.4774514
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0446062, 0.0446082
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1190782, 0.1190291
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816692, 0.0816693
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1207174, 0.1206595
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1363119, 0.1363140
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895307, 0.3895274
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3434045, 0.3434340
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3996011, 0.3996701

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2586

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265446, upper bound: 0.0265578
time: 9.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265453, upper bound: 0.0265612
time: 7.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3918255, 0.3916764
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4774867, 0.4772285
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0446099, 0.0446045
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1190273, 0.1190800
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816693, 0.0816692
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1206543, 0.1207226
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1363218, 0.1363041
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895246, 0.3895334
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3434511, 0.3433874
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3996713, 0.3996000

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2586

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265379, upper bound: 0.0265640
time: 73.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265377, upper bound: 0.0265640
time: 18.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3916771, 0.3918147
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4772289, 0.4774710
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0446050, 0.0446091
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1190798, 0.1190262
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816692, 0.0816693
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1207225, 0.1206530
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1363057, 0.1363184
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895319, 0.3895253
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3433836, 0.3434429
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3995925, 0.3996691

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2586

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265481, upper bound: 0.0265463
time: 38.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265476, upper bound: 0.0265467
time: 18.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3918020, 0.3916898
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4774519, 0.4772481
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0446086, 0.0446054
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1190289, 0.1190771
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816693, 0.0816692
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1206594, 0.1207161
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1363156, 0.1363085
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895259, 0.3895313
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3434302, 0.3433962
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3996627, 0.3995988

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2586

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265418, upper bound: 0.0265447
time: 38.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265420, upper bound: 0.0265518
time: 7.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3916898, 0.3918021
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4772481, 0.4774519
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0446054, 0.0446086
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1190771, 0.1190289
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816692, 0.0816693
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1207161, 0.1206594
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1363085, 0.1363156
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895313, 0.3895259
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3433963, 0.3434301
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3995988, 0.3996627

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2586

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265510, upper bound: 0.0265435
time: 104.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265516, upper bound: 0.0265458
time: 7.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3918148, 0.3916771
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4774711, 0.4772290
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0446091, 0.0446050
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1190262, 0.1190798
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816693, 0.0816692
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1206530, 0.1207225
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1363184, 0.1363057
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895252, 0.3895320
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3434428, 0.3433836
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3996691, 0.3995926

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2586

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265448, upper bound: 0.0265469
time: 30.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265433, upper bound: 0.0265508
time: 7.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3916765, 0.3918254
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4772284, 0.4774867
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0446045, 0.0446099
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1190800, 0.1190273
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816692, 0.0816693
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1207226, 0.1206543
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1363042, 0.1363218
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895335, 0.3895246
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3433874, 0.3434511
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3995999, 0.3996713

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2586

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265628, upper bound: 0.0265398
time: 7.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265628, upper bound: 0.0265395
time: 75.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3918015, 0.3917005
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4774513, 0.4772638
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0446082, 0.0446062
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1190291, 0.1190782
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816693, 0.0816692
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1206595, 0.1207174
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1363140, 0.1363119
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895274, 0.3895307
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3434340, 0.3434045
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3996701, 0.3996012

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2586

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265568, upper bound: 0.0265460
time: 16.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265545, upper bound: 0.0265455
time: 7.60 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265446, upper bound: 0.0265578
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265453, upper bound: 0.0265612
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265379, upper bound: 0.0265640
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265377, upper bound: 0.0265640
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265481, upper bound: 0.0265463
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265476, upper bound: 0.0265467
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265418, upper bound: 0.0265447
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265420, upper bound: 0.0265518
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265510, upper bound: 0.0265435
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265516, upper bound: 0.0265458
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265448, upper bound: 0.0265469
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265433, upper bound: 0.0265508
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265628, upper bound: 0.0265398
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265628, upper bound: 0.0265395
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265568, upper bound: 0.0265460
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.49
Output dim: 7, lower bound: -0.0265545, upper bound: 0.0265455

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3902055, 0.3904456
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4744501, 0.4748685
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0444699, 0.0444815
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1183280, 0.1181985
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816633, 0.0816638
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1198040, 0.1196713
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1362623, 0.1362645
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895305, 0.3895272
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3426521, 0.3427403
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3987470, 0.3988783

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2092

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265447, upper bound: 0.0265571
time: 12.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265449, upper bound: 0.0265586
time: 8.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3903467, 0.3903064
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4746845, 0.4746376
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0444798, 0.0444719
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1182476, 0.1182837
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816638, 0.0816633
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1197292, 0.1197474
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1362623, 0.1362644
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895305, 0.3895272
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3427119, 0.3426816
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3988103, 0.3988160

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2092

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265455, upper bound: 0.0265552
time: 57.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265443, upper bound: 0.0265594
time: 77.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3903304, 0.3903226
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4746730, 0.4746492
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0444736, 0.0444781
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1182818, 0.1182495
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816634, 0.0816638
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1197422, 0.1197344
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1362722, 0.1362546
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895244, 0.3895332
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3426987, 0.3426948
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3988172, 0.3988091

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2092

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265391, upper bound: 0.0265608
time: 114.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265396, upper bound: 0.0265634
time: 85.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3904696, 0.3901815
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4749038, 0.4744147
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0444832, 0.0444682
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1181967, 0.1183299
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816638, 0.0816633
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1196661, 0.1198093
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1362722, 0.1362546
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895244, 0.3895332
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3427575, 0.3426350
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3988796, 0.3987457

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2092

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265386, upper bound: 0.0265640
time: 96.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265387, upper bound: 0.0265653
time: 7.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3901820, 0.3904589
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4744153, 0.4748882
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0444687, 0.0444824
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1183296, 0.1181956
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816633, 0.0816638
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1198092, 0.1196648
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1362561, 0.1362689
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895317, 0.3895251
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3426312, 0.3427492
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3987384, 0.3988773

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2092

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265489, upper bound: 0.0265487
time: 6.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265481, upper bound: 0.0265478
time: 204.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3903232, 0.3903198
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4746497, 0.4746574
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0444785, 0.0444728
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1182492, 0.1182807
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0816638, 0.0816633
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1197343, 0.1197409
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1362562, 0.1362688
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3895317, 0.3895251
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3426909, 0.3426904
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.3988017, 0.3988149

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3575

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2092

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265495, upper bound: 0.0265414
time: 37.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265478, upper bound: 0.0265498
time: 6.15 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 49.97 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265447, upper bound: 0.0265571
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265449, upper bound: 0.0265586
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265455, upper bound: 0.0265552
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265443, upper bound: 0.0265594
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265391, upper bound: 0.0265608
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265396, upper bound: 0.0265634
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265386, upper bound: 0.0265640
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265387, upper bound: 0.0265653
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265489, upper bound: 0.0265487
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265481, upper bound: 0.0265478
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265495, upper bound: 0.0265414
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.97
Output dim: 7, lower bound: -0.0265478, upper bound: 0.0265498
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 49.97
Output dim: 7, lower bound: -0.0265418, upper bound: 0.0265447
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 49.97
Output dim: 7, lower bound: -0.0265420, upper bound: 0.0265518
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 49.97
Output dim: 7, lower bound: -0.0265510, upper bound: 0.0265435
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 49.97
Output dim: 7, lower bound: -0.0265516, upper bound: 0.0265458
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 49.97
Output dim: 7, lower bound: -0.0265448, upper bound: 0.0265469
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 49.97
Output dim: 7, lower bound: -0.0265433, upper bound: 0.0265508
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 49.97
Output dim: 7, lower bound: -0.0265628, upper bound: 0.0265398
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 49.97
Output dim: 7, lower bound: -0.0265628, upper bound: 0.0265395
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 49.97
Output dim: 7, lower bound: -0.0265568, upper bound: 0.0265460
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 49.97
Output dim: 7, lower bound: -0.0265545, upper bound: 0.0265455

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 74.34 + 1740.09 = 1814.43 seconds
