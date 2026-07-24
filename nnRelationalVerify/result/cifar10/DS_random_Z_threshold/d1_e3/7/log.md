## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 7)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.061406136


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2169807, 0.2169807)
1: (-4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3629912, 0.3629913)
2: (-0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0410401, 0.0410401)
3: (-0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246578, 0.0246578)
4: (-0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1151670, 0.1151670)
5: (-0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861651, 0.0861651)
6: (-0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967382, 0.0967382)
7: (-0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0845230, 0.0845230)
8: (-6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3075459, 0.3075459)
9: (-4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2648686, 0.2648686)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.08 + 78.60 = 86.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0620242, upper bound: 0.0620248

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2294

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 157

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619868, upper bound: 0.0618860
time: 4.09 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618853, upper bound: 0.0619872
time: 83.29 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 87.39 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 87.39
Output dim: 1, lower bound: -0.0619868, upper bound: 0.0618860
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 87.39
Output dim: 1, lower bound: -0.0618853, upper bound: 0.0619872

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2158461, 0.2157844
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3611407, 0.3610411
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409459, 0.0409483
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246496, 0.0246500
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1139533, 0.1140057
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861388, 0.0861394
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967342, 0.0967345
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0844156, 0.0844218
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3058870, 0.3057973
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2634356, 0.2633604

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2528

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 683

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619850, upper bound: 0.0618840
time: 30.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619843, upper bound: 0.0618858
time: 3.72 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2157844, 0.2158461
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3610411, 0.3611407
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409483, 0.0409459
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246500, 0.0246496
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1140057, 0.1139533
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861394, 0.0861388
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967345, 0.0967342
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0844218, 0.0844156
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3057973, 0.3058870
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2633604, 0.2634356

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2149

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2878

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618698, upper bound: 0.0619641
time: 62.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618612, upper bound: 0.0619720
time: 29.81 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 99.04 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 99.04
Output dim: 1, lower bound: -0.0619850, upper bound: 0.0618840
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 99.04
Output dim: 1, lower bound: -0.0619843, upper bound: 0.0618858
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 99.04
Output dim: 1, lower bound: -0.0618698, upper bound: 0.0619641
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 99.04
Output dim: 1, lower bound: -0.0618612, upper bound: 0.0619720

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2158456, 0.2157842
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3611441, 0.3610401
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409454, 0.0409492
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246497, 0.0246500
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1139526, 0.1140084
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861387, 0.0861394
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967339, 0.0967358
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0844155, 0.0844218
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3058875, 0.3057970
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2634418, 0.2633590

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 759

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619754, upper bound: 0.0616288
time: 144.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0617312, upper bound: 0.0618743
time: 6.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2158461, 0.2157840
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3611399, 0.3610411
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409459, 0.0409479
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246497, 0.0246500
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1139533, 0.1140050
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861388, 0.0861394
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967342, 0.0967342
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0844155, 0.0844218
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3058866, 0.3057973
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2634342, 0.2633604

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2983

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 679

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619830, upper bound: 0.0618823
time: 124.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619833, upper bound: 0.0618819
time: 120.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2158130, 0.2158387
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3610908, 0.3611298
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409531, 0.0409447
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246494, 0.0246490
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1140053, 0.1139532
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861408, 0.0861381
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967343, 0.0967335
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0844194, 0.0844237
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3058367, 0.3058753
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2633835, 0.2634279

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2149

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2194

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618319, upper bound: 0.0613675
time: 180.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0613755, upper bound: 0.0619260
time: 111.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2157770, 0.2158461
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3610301, 0.3611407
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409471, 0.0409459
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246493, 0.0246496
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1140057, 0.1139529
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861387, 0.0861388
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967338, 0.0967342
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0844218, 0.0844132
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3057857, 0.3058870
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2633526, 0.2634356

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 678

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2285

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618530, upper bound: 0.0619522
time: 60.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618414, upper bound: 0.0619645
time: 5.23 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 72.40 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 72.40
Output dim: 1, lower bound: -0.0619754, upper bound: 0.0616288
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 72.40
Output dim: 1, lower bound: -0.0617312, upper bound: 0.0618743
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 72.40
Output dim: 1, lower bound: -0.0619830, upper bound: 0.0618823
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 72.40
Output dim: 1, lower bound: -0.0619833, upper bound: 0.0618819
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 72.40
Output dim: 1, lower bound: -0.0618319, upper bound: 0.0613675
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 72.40
Output dim: 1, lower bound: -0.0613755, upper bound: 0.0619260
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 72.40
Output dim: 1, lower bound: -0.0618530, upper bound: 0.0619522
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 72.40
Output dim: 1, lower bound: -0.0618414, upper bound: 0.0619645

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2131801, 0.2130351
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3560941, 0.3558317
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409051, 0.0409090
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0245366, 0.0245406
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1138090, 0.1138681
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0860841, 0.0860868
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0959961, 0.0960010
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0842299, 0.0842423
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3046615, 0.3045317
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2620313, 0.2619048

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 773

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2151

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619557, upper bound: 0.0608539
time: 93.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0612013, upper bound: 0.0616088
time: 36.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2130966, 0.2131186
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3559357, 0.3559905
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409052, 0.0409089
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0245403, 0.0245369
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1138123, 0.1138648
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0860860, 0.0860848
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0959991, 0.0959980
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0842362, 0.0842361
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3046223, 0.3045709
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2619877, 0.2619485

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2151

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2543

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0613887, upper bound: 0.0614868
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0613887, upper bound: 0.0614868
time: 3.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2158459, 0.2157838
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3611341, 0.3610361
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409453, 0.0409470
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246495, 0.0246498
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1139533, 0.1140050
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861385, 0.0861392
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967335, 0.0967334
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0844150, 0.0844214
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3058852, 0.3057960
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2634273, 0.2633535

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2983

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 718

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619825, upper bound: 0.0618817
time: 49.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619822, upper bound: 0.0618813
time: 37.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2158459, 0.2157838
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3611349, 0.3610353
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409449, 0.0409474
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246495, 0.0246499
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1139533, 0.1140050
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861385, 0.0861392
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967334, 0.0967334
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0844151, 0.0844213
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3058854, 0.3057958
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2634274, 0.2633533

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 785

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2158

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619728, upper bound: 0.0617565
time: 84.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618567, upper bound: 0.0618739
time: 3.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2157143, 0.2156045
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3608253, 0.3604274
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409527, 0.0409446
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246493, 0.0246489
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1138772, 0.1138969
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861264, 0.0861331
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967337, 0.0967333
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0843877, 0.0844036
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3056110, 0.3053067
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2631482, 0.2627796

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 771

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0615930, upper bound: 0.0609525
time: 87.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0613147, upper bound: 0.0612296
time: 220.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2155790, 0.2157400
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3603885, 0.3608642
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409530, 0.0409442
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246493, 0.0246489
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1139490, 0.1138250
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861358, 0.0861236
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967341, 0.0967329
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0843993, 0.0843920
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3052680, 0.3056496
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2627352, 0.2631924

Time for backsubstitution: 6.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2340

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 755

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0613567, upper bound: 0.0617838
time: 38.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0612333, upper bound: 0.0619071
time: 8.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2157726, 0.2158400
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3607709, 0.3608723
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409021, 0.0409014
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246468, 0.0246471
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1140050, 0.1139526
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861384, 0.0861383
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0967063, 0.0967076
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0844158, 0.0844070
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3057652, 0.3058639
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2630758, 0.2631521

Time for backsubstitution: 6.20 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 86.69 + 1716.01 = 1802.69 seconds
