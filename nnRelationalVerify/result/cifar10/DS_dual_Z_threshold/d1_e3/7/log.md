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
execution time: IAR + RelationalAnalysis = 7.88 + 79.77 = 87.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0620242, upper bound: 0.0620248

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2634

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0617831, upper bound: 0.0618663
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618643, upper bound: 0.0617844
time: 89.19 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 93.71 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 93.71
Output dim: 1, lower bound: -0.0617831, upper bound: 0.0618663
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 93.71
Output dim: 1, lower bound: -0.0618643, upper bound: 0.0617844

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2146671, 0.2146809
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3565923, 0.3566599
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0410987, 0.0411039
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246250, 0.0246246
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1120378, 0.1119506
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861070, 0.0861070
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0966903, 0.0966895
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0845143, 0.0845144
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3030078, 0.3030812
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2603464, 0.2604132

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0613560, upper bound: 0.0613565
time: 382.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0616510, upper bound: 0.0614383
time: 19.79 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2146809, 0.2146671
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3566598, 0.3565922
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0411038, 0.0410987
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0246246, 0.0246250
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1119506, 0.1120378
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0861070, 0.0861070
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0966895, 0.0966903
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0845145, 0.0845143
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3030812, 0.3030078
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2604132, 0.2603464

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0614372, upper bound: 0.0616532
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0617322, upper bound: 0.0613573
time: 49.70 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 59.30 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 59.30
Output dim: 1, lower bound: -0.0613560, upper bound: 0.0613565
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 59.30
Output dim: 1, lower bound: -0.0616510, upper bound: 0.0614383
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 59.30
Output dim: 1, lower bound: -0.0614372, upper bound: 0.0616532
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 59.30
Output dim: 1, lower bound: -0.0617322, upper bound: 0.0613573

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2121180, 0.2119929
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3531959, 0.3530124
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409982, 0.0410070
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0244980, 0.0245065
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1119038, 0.1118254
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0860022, 0.0860077
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0966482, 0.0966473
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0842001, 0.0842155
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.2999636, 0.2998526
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2581885, 0.2581037

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0615755, upper bound: 0.0611694
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0613603, upper bound: 0.0613627
time: 12.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2119928, 0.2121180
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3530123, 0.3531960
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0410071, 0.0409982
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0245064, 0.0244980
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1118254, 0.1119037
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0860077, 0.0860022
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0966473, 0.0966482
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0842156, 0.0842000
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.2998526, 0.2999636
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2581037, 0.2581885

Time for backsubstitution: 6.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0613620, upper bound: 0.0613609
time: 110.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0611682, upper bound: 0.0615763
time: 16.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2121318, 0.2119790
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3532634, 0.3529447
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0410033, 0.0410019
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0244976, 0.0245068
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1118165, 0.1119126
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0860022, 0.0860077
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0966475, 0.0966480
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0842003, 0.0842154
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.3000368, 0.2997793
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2582551, 0.2580370

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0616574, upper bound: 0.0610667
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0614640, upper bound: 0.0612822
time: 3.75 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 13.33 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.33
Output dim: 1, lower bound: -0.0615755, upper bound: 0.0611694
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 13.33
Output dim: 1, lower bound: -0.0613603, upper bound: 0.0613627
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 13.33
Output dim: 1, lower bound: -0.0613620, upper bound: 0.0613609
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.33
Output dim: 1, lower bound: -0.0611682, upper bound: 0.0615763
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.33
Output dim: 1, lower bound: -0.0616574, upper bound: 0.0610667
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.33
Output dim: 1, lower bound: -0.0614640, upper bound: 0.0612822

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2104584, 0.2102895
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3483396, 0.3480293
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0410159, 0.0410186
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0244917, 0.0245003
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1090216, 0.1090124
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0859300, 0.0859368
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0966050, 0.0966054
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0841272, 0.0841417
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.2960271, 0.2958189
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2535112, 0.2532902

Time for backsubstitution: 6.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2619

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0613629, upper bound: 0.0610638
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0614132, upper bound: 0.0608597
time: 3.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2102894, 0.2104583
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3480294, 0.3483396
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0410186, 0.0410158
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0245003, 0.0244917
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1090123, 0.1090216
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0859368, 0.0859300
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0966054, 0.0966051
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0841416, 0.0841272
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.2958189, 0.2960271
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2532902, 0.2535112

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2619

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0608584, upper bound: 0.0614141
time: 32.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0610638, upper bound: 0.0613630
time: 12.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2104648, 0.2102757
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3483942, 0.3479618
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0410201, 0.0410135
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0244914, 0.0245007
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1089344, 0.1090658
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0859299, 0.0859364
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0966043, 0.0966058
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0841274, 0.0841415
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.2960825, 0.2957456
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2535506, 0.2532234

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2619

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0613878, upper bound: 0.0609044
time: 19.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0615220, upper bound: 0.0608363
time: 6.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2104284, 0.2103195
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3482804, 0.3480885
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0410148, 0.0410196
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0244914, 0.0245006
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1090035, 0.1090305
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0859313, 0.0859355
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0966056, 0.0966049
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0841263, 0.0841426
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.2960032, 0.2958429
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2534416, 0.2533597

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2619

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0611632, upper bound: 0.0611189
time: 122.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0613587, upper bound: 0.0610607
time: 27.51 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 155.84 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 155.84
Output dim: 1, lower bound: -0.0613629, upper bound: 0.0610638
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 155.84
Output dim: 1, lower bound: -0.0614132, upper bound: 0.0608597
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 155.84
Output dim: 1, lower bound: -0.0608584, upper bound: 0.0614141
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 155.84
Output dim: 1, lower bound: -0.0610638, upper bound: 0.0613630
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 155.84
Output dim: 1, lower bound: -0.0613878, upper bound: 0.0609044
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 155.84
Output dim: 1, lower bound: -0.0615220, upper bound: 0.0608363
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 155.84
Output dim: 1, lower bound: -0.0611632, upper bound: 0.0611189
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 155.84
Output dim: 1, lower bound: -0.0613587, upper bound: 0.0610607

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2055409, 0.2053448
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3417256, 0.3413707
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0410186, 0.0410213
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0243202, 0.0243301
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1073914, 0.1073964
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0858318, 0.0858389
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0962683, 0.0962704
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0837773, 0.0837952
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.2928376, 0.2926226
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2518623, 0.2516370

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0613539, upper bound: 0.0606291
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0611823, upper bound: 0.0607992
time: 50.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2053448, 0.2055409
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3413707, 0.3417256
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0410213, 0.0410185
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0243301, 0.0243202
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1073964, 0.1073914
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0858389, 0.0858317
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0962704, 0.0962683
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0837953, 0.0837772
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.2926226, 0.2928376
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2516370, 0.2518623

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0607990, upper bound: 0.0606036
time: 298.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0606275, upper bound: 0.0613555
time: 3.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2055742, 0.2053310
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3418282, 0.3413032
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0410228, 0.0410162
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0243198, 0.0243307
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1073042, 0.1074690
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0858317, 0.0858391
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0962676, 0.0962716
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0837775, 0.0837952
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.2929176, 0.2925493
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2519164, 0.2515702

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0614627, upper bound: 0.0606044
time: 220.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0612911, upper bound: 0.0607766
time: 6.35 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 231.91 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 231.91
Output dim: 1, lower bound: -0.0613539, upper bound: 0.0606291
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 231.91
Output dim: 1, lower bound: -0.0611823, upper bound: 0.0607992
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 231.91
Output dim: 1, lower bound: -0.0607990, upper bound: 0.0606036
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 231.91
Output dim: 1, lower bound: -0.0606275, upper bound: 0.0613555
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 231.91
Output dim: 1, lower bound: -0.0614627, upper bound: 0.0606044
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 231.91
Output dim: 1, lower bound: -0.0612911, upper bound: 0.0607766

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1120472, -4.3782978, -5.1120472, -4.3782978, -0.2040428, 0.2037807
1: -4.7961903, -3.5826273, -4.7961903, -3.5826273, -0.3365631, 0.3359648
2: -0.8544037, -0.6336883, -0.8544037, -0.6336883, -0.0409071, 0.0408967
3: -0.0635773, 0.0526330, -0.0635773, 0.0526330, -0.0243165, 0.0243274
4: -0.4156742, -0.0396947, -0.4156742, -0.0396947, -0.1054833, 0.1056734
5: -0.2217601, -0.1036188, -0.2217601, -0.1036188, -0.0857505, 0.0857583
6: -0.7439466, -0.5441107, -0.7439466, -0.5441107, -0.0962293, 0.0962336
7: -0.4693450, -0.2704960, -0.4693450, -0.2704960, -0.0837305, 0.0837480
8: -6.7065873, -5.8053002, -6.7065873, -5.8053002, -0.2889448, 0.2885199
9: -4.3003297, -3.4784088, -4.3003297, -3.4784088, -0.2473502, 0.2469398

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2637

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0612984, upper bound: 0.0605426
time: 31.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0613992, upper bound: 0.0604407
time: 243.72 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 280.84 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 280.84
Output dim: 1, lower bound: -0.0612984, upper bound: 0.0605426
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 280.84
Output dim: 1, lower bound: -0.0613992, upper bound: 0.0604407

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 87.65 + 1862.75 = 1950.39 seconds
