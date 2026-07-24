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
execution time: IAR + RelationalAnalysis = 8.19 + 65.89 = 74.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0265680, upper bound: 0.0265689

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3329

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2529

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265579, upper bound: 0.0265694
time: 7.52 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265674, upper bound: 0.0265576
time: 81.81 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 89.34 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 89.34
Output dim: 7, lower bound: -0.0265579, upper bound: 0.0265694
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 89.34
Output dim: 7, lower bound: -0.0265674, upper bound: 0.0265576

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3976601, 0.3976601
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4870203, 0.4870241
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0448761, 0.0448753
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210925, 0.1210908
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815625, 0.0815637
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1236142, 0.1236137
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355916, 0.1355913
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3894928, 0.3894942
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3473954, 0.3474012
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4040843, 0.4040877

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2131

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265568, upper bound: 0.0265708
time: 6.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265568, upper bound: 0.0265682
time: 12.55 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3976602, 0.3976601
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4870240, 0.4870203
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0448753, 0.0448761
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210908, 0.1210925
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815637, 0.0815625
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1236137, 0.1236142
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355913, 0.1355916
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3894942, 0.3894928
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3474012, 0.3473954
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4040877, 0.4040843

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2833

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2718

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265644, upper bound: 0.0265487
time: 19.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265567, upper bound: 0.0265604
time: 6.65 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.79 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.79
Output dim: 7, lower bound: -0.0265568, upper bound: 0.0265708
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.79
Output dim: 7, lower bound: -0.0265568, upper bound: 0.0265682
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.79
Output dim: 7, lower bound: -0.0265644, upper bound: 0.0265487
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.79
Output dim: 7, lower bound: -0.0265567, upper bound: 0.0265604

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3976226, 0.3976046
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4870102, 0.4870086
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0448553, 0.0448589
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210894, 0.1210886
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815619, 0.0815633
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1236079, 0.1236077
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355881, 0.1355888
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3894928, 0.3894941
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3473276, 0.3472791
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4040630, 0.4040518

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3327

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3221

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265507, upper bound: 0.0265670
time: 186.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265565, upper bound: 0.0265596
time: 376.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3976046, 0.3976226
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4870049, 0.4870139
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0448597, 0.0448546
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210903, 0.1210877
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815620, 0.0815632
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1236082, 0.1236073
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355890, 0.1355878
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3894928, 0.3894941
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3472733, 0.3473334
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4040483, 0.4040665

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 704

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2745

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265580, upper bound: 0.0265646
time: 6.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265520, upper bound: 0.0265685
time: 9.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3974037, 0.3974019
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4868558, 0.4868460
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0441758, 0.0442020
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209842, 0.1209865
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815352, 0.0815342
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1232757, 0.1232803
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1351555, 0.1351759
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3891522, 0.3891420
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3463014, 0.3462754
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4039170, 0.4038762

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2541

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2849

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265658, upper bound: 0.0265514
time: 38.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265658, upper bound: 0.0265498
time: 40.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3974019, 0.3974037
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4868497, 0.4868521
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0442013, 0.0441765
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209848, 0.1209859
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815355, 0.0815340
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1232798, 0.1232762
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1351756, 0.1351558
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3891434, 0.3891509
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3462812, 0.3462956
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4038796, 0.4039136

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2334

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2929

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265573, upper bound: 0.0265568
time: 40.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265592, upper bound: 0.0265584
time: 5.96 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 52.37 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 52.37
Output dim: 7, lower bound: -0.0265507, upper bound: 0.0265670
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 52.37
Output dim: 7, lower bound: -0.0265565, upper bound: 0.0265596
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 52.37
Output dim: 7, lower bound: -0.0265580, upper bound: 0.0265646
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 52.37
Output dim: 7, lower bound: -0.0265520, upper bound: 0.0265685
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 52.37
Output dim: 7, lower bound: -0.0265658, upper bound: 0.0265514
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 52.37
Output dim: 7, lower bound: -0.0265658, upper bound: 0.0265498
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 52.37
Output dim: 7, lower bound: -0.0265573, upper bound: 0.0265568
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 52.37
Output dim: 7, lower bound: -0.0265592, upper bound: 0.0265584

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3972777, 0.3972547
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4868755, 0.4868764
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0447970, 0.0447992
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209897, 0.1209832
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815188, 0.0815207
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1234558, 0.1234456
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355842, 0.1355861
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3897644, 0.3897569
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3472110, 0.3471583
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4036879, 0.4036733

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265509, upper bound: 0.0265701
time: 6.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265509, upper bound: 0.0265708
time: 20.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3972727, 0.3972597
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4868781, 0.4868738
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0447956, 0.0448006
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209840, 0.1209890
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815193, 0.0815201
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1234458, 0.1234556
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355854, 0.1355849
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3897555, 0.3897656
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3472068, 0.3471625
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4036846, 0.4036766

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2839

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2642

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265501, upper bound: 0.0265433
time: 6.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265364, upper bound: 0.0265587
time: 15.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3975625, 0.3975682
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4869857, 0.4869913
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0447988, 0.0448046
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210890, 0.1210863
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815534, 0.0815551
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1235917, 0.1235940
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355448, 0.1355526
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3894737, 0.3894708
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3471471, 0.3471738
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4039826, 0.4039810

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2831

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3044

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265561, upper bound: 0.0265640
time: 63.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265561, upper bound: 0.0265636
time: 58.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3975502, 0.3975805
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4869823, 0.4869946
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0448097, 0.0447938
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210889, 0.1210864
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815539, 0.0815546
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1235949, 0.1235908
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355538, 0.1355436
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3894694, 0.3894750
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3471138, 0.3472072
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4039629, 0.4040008

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2704

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 655

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265521, upper bound: 0.0265671
time: 76.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265542, upper bound: 0.0265714
time: 6.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3974037, 0.3974019
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4868558, 0.4868460
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0441758, 0.0442020
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209842, 0.1209865
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815352, 0.0815342
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1232757, 0.1232803
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1351555, 0.1351759
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3891522, 0.3891420
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3463014, 0.3462754
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4039170, 0.4038762

Time for backsubstitution: 6.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 722

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2937

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265654, upper bound: 0.0265509
time: 18.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265660, upper bound: 0.0265458
time: 24.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3974037, 0.3974019
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4868558, 0.4868460
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0441758, 0.0442020
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209842, 0.1209865
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815352, 0.0815342
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1232757, 0.1232803
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1351555, 0.1351759
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3891522, 0.3891420
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3463014, 0.3462754
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4039170, 0.4038762

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3014

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3124

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265500, upper bound: 0.0265532
time: 21.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265641, upper bound: 0.0265350
time: 46.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3973917, 0.3973938
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4868497, 0.4868520
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0441966, 0.0441720
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209756, 0.1209762
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815357, 0.0815342
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1232790, 0.1232755
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1351758, 0.1351561
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3891469, 0.3891544
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3462810, 0.3462954
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4038796, 0.4039136

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3086

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 654

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265580, upper bound: 0.0265568
time: 7.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265585, upper bound: 0.0265594
time: 6.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3973920, 0.3973936
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4868497, 0.4868520
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0441968, 0.0441719
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209752, 0.1209766
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815357, 0.0815341
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1232790, 0.1232755
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1351759, 0.1351560
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3891470, 0.3891543
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3462810, 0.3462954
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4038796, 0.4039136

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3304

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2616

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265513, upper bound: 0.0265456
time: 8.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265449, upper bound: 0.0265511
time: 29.93 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 44.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265509, upper bound: 0.0265701
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265509, upper bound: 0.0265708
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265501, upper bound: 0.0265433
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265364, upper bound: 0.0265587
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265561, upper bound: 0.0265640
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265561, upper bound: 0.0265636
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265521, upper bound: 0.0265671
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265542, upper bound: 0.0265714
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265654, upper bound: 0.0265509
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265660, upper bound: 0.0265458
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265500, upper bound: 0.0265532
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265641, upper bound: 0.0265350
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265580, upper bound: 0.0265568
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265585, upper bound: 0.0265594
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265513, upper bound: 0.0265456
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.60
Output dim: 7, lower bound: -0.0265449, upper bound: 0.0265511

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3972777, 0.3972547
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4868755, 0.4868764
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0447970, 0.0447992
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209897, 0.1209832
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815188, 0.0815207
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1234558, 0.1234456
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355842, 0.1355861
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3897644, 0.3897569
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3472110, 0.3471583
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4036879, 0.4036733

Time for backsubstitution: 6.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2572

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3363

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265524, upper bound: 0.0265718
time: 6.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265505, upper bound: 0.0265700
time: 7.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3972777, 0.3972547
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4868755, 0.4868764
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0447970, 0.0447992
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209897, 0.1209832
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815188, 0.0815207
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1234558, 0.1234456
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355842, 0.1355861
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3897644, 0.3897569
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3472110, 0.3471583
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4036879, 0.4036733

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2556

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2684

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265525, upper bound: 0.0265711
time: 9.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265525, upper bound: 0.0265698
time: 18.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3963367, 0.3963959
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4848468, 0.4849843
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0447627, 0.0447673
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209413, 0.1209469
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0814850, 0.0814805
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1233726, 0.1233782
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1354684, 0.1354673
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3897250, 0.3897347
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3442946, 0.3444328
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4014539, 0.4016207

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2586

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3253

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265489, upper bound: 0.0265405
time: 42.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265470, upper bound: 0.0265377
time: 33.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3964087, 0.3963239
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4849885, 0.4848426
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0447623, 0.0447677
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1209419, 0.1209462
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0814797, 0.0814858
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1233684, 0.1233824
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1354678, 0.1354679
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3897246, 0.3897351
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3444770, 0.3442504
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4016287, 0.4014459

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2684

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265371, upper bound: 0.0265563
time: 99.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0265371, upper bound: 0.0265432
time: 62.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3975625, 0.3975682
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4869857, 0.4869913
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0447988, 0.0448046
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210890, 0.1210863
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815534, 0.0815551
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1235917, 0.1235940
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355448, 0.1355526
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3894737, 0.3894708
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3471471, 0.3471738
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4039826, 0.4039810

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2679

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 756

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265567, upper bound: 0.0265659
time: 26.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265552, upper bound: 0.0265613
time: 8.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.2564082, -3.5876958, -4.2564082, -3.5876958, -0.3975625, 0.3975682
1: -5.9149027, -4.9843111, -5.9149027, -4.9843111, -0.4869857, 0.4869913
2: -0.6954473, -0.4670191, -0.6954473, -0.4670191, -0.0447988, 0.0448046
3: -0.9623526, -0.5934623, -0.9623526, -0.5934623, -0.1210890, 0.1210863
4: -0.2890245, -0.0123321, -0.2890245, -0.0123321, -0.0815534, 0.0815551
5: -1.0052335, -0.6736995, -1.0052335, -0.6736995, -0.1235917, 0.1235940
6: -1.0789576, -0.4399564, -1.0789576, -0.4399564, -0.1355448, 0.1355526
7: -0.5497090, -0.0507912, -0.5497090, -0.0507912, -0.3894737, 0.3894708
8: -5.4932861, -4.6516600, -5.4932861, -4.6516600, -0.3471471, 0.3471738
9: -4.9619889, -4.2091098, -4.9619889, -4.2091098, -0.4039826, 0.4039810

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2760
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2833
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 655
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3297
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 727

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265572, upper bound: 0.0265520
time: 62.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265570, upper bound: 0.0265640
time: 21.21 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 90.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265524, upper bound: 0.0265718
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265505, upper bound: 0.0265700
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265525, upper bound: 0.0265711
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265525, upper bound: 0.0265698
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265489, upper bound: 0.0265405
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265470, upper bound: 0.0265377
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265371, upper bound: 0.0265563
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265371, upper bound: 0.0265432
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265567, upper bound: 0.0265659
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265552, upper bound: 0.0265613
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265572, upper bound: 0.0265520
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 90.41
Output dim: 7, lower bound: -0.0265570, upper bound: 0.0265640
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 90.41
Output dim: 7, lower bound: -0.0265521, upper bound: 0.0265671
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 90.41
Output dim: 7, lower bound: -0.0265542, upper bound: 0.0265714
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 90.41
Output dim: 7, lower bound: -0.0265654, upper bound: 0.0265509
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 90.41
Output dim: 7, lower bound: -0.0265660, upper bound: 0.0265458
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 90.41
Output dim: 7, lower bound: -0.0265500, upper bound: 0.0265532
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 90.41
Output dim: 7, lower bound: -0.0265641, upper bound: 0.0265350
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 90.41
Output dim: 7, lower bound: -0.0265580, upper bound: 0.0265568
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 90.41
Output dim: 7, lower bound: -0.0265585, upper bound: 0.0265594
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 90.41
Output dim: 7, lower bound: -0.0265513, upper bound: 0.0265456
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 90.41
Output dim: 7, lower bound: -0.0265449, upper bound: 0.0265511

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 74.08 + 1780.58 = 1854.66 seconds
