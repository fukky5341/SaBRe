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
execution time: IAR + RelationalAnalysis = 7.80 + 77.11 = 84.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0620242, upper bound: 0.0620248

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2752
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2747
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 2933
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3373

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3036

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0620209, upper bound: 0.0610954
time: 168.41 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0620210, upper bound: 0.0620232
time: 2.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 171.07 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 171.07
Output dim: 1, lower bound: -0.0620209, upper bound: 0.0610954
NS_A2, status: Status.UNKNOWN, split count: 1, time: 171.07
Output dim: 1, lower bound: -0.0620210, upper bound: 0.0620232

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5.1120453, -4.3801327, -5.1120453, -4.3797159, -0.2149008, 0.2142887
1: -4.7961907, -3.5864530, -4.7961907, -3.5855851, -0.3586524, 0.3573756
2: -0.8544025, -0.6337172, -0.8544030, -0.6337106, -0.0410128, 0.0410052
3: -0.0635500, 0.0526329, -0.0635562, 0.0526330, -0.0246303, 0.0246365
4: -0.4153068, -0.0396948, -0.4153906, -0.0396947, -0.1146410, 0.1147600
5: -0.2217290, -0.1036194, -0.2217360, -0.1036193, -0.0861288, 0.0861367
6: -0.7428808, -0.5441110, -0.7431108, -0.5441110, -0.0956014, 0.0958313
7: -0.4692027, -0.2704975, -0.4692350, -0.2704973, -0.0843750, 0.0844080
8: -6.7065873, -5.8060899, -6.7065883, -5.8059101, -0.3066498, 0.3063865
9: -4.3003283, -3.4794269, -4.3003283, -3.4791956, -0.2637134, 0.2633736

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 2933
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3062

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0613469, upper bound: 0.0610846
time: 218.61 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0620096, upper bound: 0.0610836
time: 27.46 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5.1219254, -4.3801904, -5.1120462, -4.3801861, -0.2299936, 0.2146057
1: -4.8170094, -3.5863571, -4.7961907, -3.5863461, -0.3900863, 0.3580530
2: -0.8545897, -0.6337083, -0.8544023, -0.6337063, -0.0412215, 0.0410096
3: -0.0635209, 0.0527530, -0.0635284, 0.0526330, -0.0246388, 0.0248072
4: -0.4153428, -0.0375012, -0.4153255, -0.0396952, -0.1147290, 0.1176472
5: -0.2217119, -0.1033313, -0.2216983, -0.1036190, -0.0861872, 0.0863822
6: -0.7438121, -0.5405401, -0.7437140, -0.5441113, -0.0959597, 0.1000532
7: -0.4690571, -0.2694548, -0.4690840, -0.2704966, -0.0844839, 0.0852934
8: -6.7100005, -5.8071065, -6.7065878, -5.8067508, -0.3131382, 0.3064677
9: -4.3057861, -3.4798751, -4.3003278, -3.4797826, -0.2719170, 0.2635276

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 2933
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3373

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3062

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0613471, upper bound: 0.0620092
time: 156.03 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0620095, upper bound: 0.0620093
time: 36.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 198.06 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 198.06
Output dim: 1, lower bound: -0.0613469, upper bound: 0.0610846
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 198.06
Output dim: 1, lower bound: -0.0620096, upper bound: 0.0610836
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 198.06
Output dim: 1, lower bound: -0.0613471, upper bound: 0.0620092
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 198.06
Output dim: 1, lower bound: -0.0620095, upper bound: 0.0620093

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -5.1120453, -4.3814888, -5.1155581, -4.3812070, -0.2113667, 0.2206397
1: -4.7961907, -3.5866752, -4.8078332, -3.5858283, -0.3522959, 0.3694561
2: -0.8543852, -0.6337172, -0.8543838, -0.6332954, -0.0414719, 0.0407621
3: -0.0635370, 0.0526328, -0.0635418, 0.0526481, -0.0246610, 0.0246141
4: -0.4152923, -0.0396954, -0.4153761, -0.0393726, -0.1149813, 0.1145777
5: -0.2217125, -0.1036196, -0.2217177, -0.1032552, -0.0864686, 0.0859344
6: -0.7427617, -0.5441129, -0.7429923, -0.5438561, -0.0956581, 0.0955071
7: -0.4691698, -0.2704980, -0.4691998, -0.2694588, -0.0854534, 0.0838311
8: -6.7065873, -5.8061914, -6.7154717, -5.8060160, -0.3009460, 0.3170012
9: -4.3003278, -3.4794669, -4.3108525, -3.4792409, -0.2581631, 0.2739582

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2752
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2747
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 2933
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3373

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2149

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619752, upper bound: 0.0601784
time: 35.43 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619752, upper bound: 0.0610493
time: 21.04 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.1219244, -4.3823524, -5.1120434, -4.3826928, -0.2263100, 0.2114270
1: -4.8170094, -3.5922260, -4.7961907, -3.5931506, -0.3831205, 0.3520422
2: -0.8543704, -0.6337084, -0.8541547, -0.6337063, -0.0409902, 0.0407421
3: -0.0635086, 0.0527525, -0.0635142, 0.0526323, -0.0246169, 0.0247828
4: -0.4151717, -0.0375012, -0.4151313, -0.0396956, -0.1145536, 0.1174445
5: -0.2215196, -0.1033319, -0.2214776, -0.1036198, -0.0859878, 0.0861558
6: -0.7434750, -0.5405427, -0.7433259, -0.5441140, -0.0956168, 0.0996625
7: -0.4685165, -0.2694556, -0.4684582, -0.2704978, -0.0839257, 0.0846536
8: -6.7099991, -5.8115392, -6.7065878, -5.8118091, -0.3069849, 0.3011595
9: -4.3057861, -3.4851437, -4.3003278, -3.4858899, -0.2658095, 0.2582535

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2752
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2747
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 2933
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3373

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2149

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0613128, upper bound: 0.0611047
time: 143.90 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0613127, upper bound: 0.0619755
time: 111.96 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.1219258, -4.3815470, -5.1155596, -4.3816776, -0.2264596, 0.2209564
1: -4.8170094, -3.5865779, -4.8078332, -3.5865898, -0.3837299, 0.3701327
2: -0.8545724, -0.6337081, -0.8543838, -0.6332911, -0.0416805, 0.0407666
3: -0.0635080, 0.0527529, -0.0635139, 0.0526482, -0.0246695, 0.0247848
4: -0.4153284, -0.0375011, -0.4153117, -0.0393731, -0.1150693, 0.1174651
5: -0.2216954, -0.1033316, -0.2216801, -0.1032551, -0.0865270, 0.0861799
6: -0.7436924, -0.5405420, -0.7435966, -0.5438564, -0.0960162, 0.0997291
7: -0.4690241, -0.2694556, -0.4690486, -0.2694582, -0.0855622, 0.0847166
8: -6.7099996, -5.8072090, -6.7154717, -5.8068562, -0.3074342, 0.3170823
9: -4.3057861, -3.4799156, -4.3108521, -3.4798279, -0.2663664, 0.2741116

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2752
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2747
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 2933
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2149

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619752, upper bound: 0.0611036
time: 425.66 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0619753, upper bound: 0.0619751
time: 49.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 481.59 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 481.59
Output dim: 1, lower bound: -0.0619752, upper bound: 0.0601784
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 481.59
Output dim: 1, lower bound: -0.0619752, upper bound: 0.0610493
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 481.59
Output dim: 1, lower bound: -0.0613128, upper bound: 0.0611047
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 481.59
Output dim: 1, lower bound: -0.0613127, upper bound: 0.0619755
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 481.59
Output dim: 1, lower bound: -0.0619752, upper bound: 0.0611036
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 481.59
Output dim: 1, lower bound: -0.0619753, upper bound: 0.0619751

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.1120453, -4.3845348, -5.1155581, -4.3838263, -0.2087188, 0.2175828
1: -4.7961903, -3.5925384, -4.8078337, -3.5908694, -0.3472629, 0.3636439
2: -0.8543817, -0.6337172, -0.8543813, -0.6332954, -0.0414677, 0.0407586
3: -0.0635051, 0.0526328, -0.0635142, 0.0526480, -0.0246293, 0.0245864
4: -0.4147793, -0.0396951, -0.4149333, -0.0393727, -0.1144845, 0.1141377
5: -0.2215770, -0.1036210, -0.2216013, -0.1032564, -0.0863396, 0.0858201
6: -0.7427437, -0.5441128, -0.7429767, -0.5438559, -0.0956285, 0.0954816
7: -0.4686994, -0.2704982, -0.4687946, -0.2694588, -0.0850223, 0.0834550
8: -6.7065816, -5.8094225, -6.7154665, -5.8087926, -0.2982342, 0.3138707
9: -4.3003278, -3.4831357, -4.3108525, -3.4823947, -0.2550049, 0.2703068

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 2933
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2605

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618432, upper bound: 0.0597505
time: 145.21 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618432, upper bound: 0.0600462
time: 3.67 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.1176267, -4.3818178, -5.1155577, -4.3814974, -0.2166532, 0.2178737
1: -4.8067060, -3.5874357, -4.8078341, -3.5864973, -0.3622981, 0.3642018
2: -0.8543814, -0.6337140, -0.8543793, -0.6332954, -0.0414706, 0.0407653
3: -0.0635267, 0.0526998, -0.0635327, 0.0526478, -0.0246319, 0.0246750
4: -0.4152828, -0.0387475, -0.4153281, -0.0393730, -0.1145875, 0.1154384
5: -0.2216772, -0.1033310, -0.2216868, -0.1032557, -0.0863539, 0.0861548
6: -0.7427372, -0.5441031, -0.7429702, -0.5438558, -0.0956356, 0.0955286
7: -0.4691035, -0.2696353, -0.4691418, -0.2694589, -0.0850681, 0.0845764
8: -6.7125053, -5.8064837, -6.7154689, -5.8062687, -0.3064325, 0.3141798
9: -4.3069649, -3.4799042, -4.3108530, -3.4796281, -0.2644544, 0.2706611

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 2933
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2605

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618433, upper bound: 0.0606236
time: 166.14 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0618430, upper bound: 0.0609171
time: 4.12 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 84.91 + 1758.42 = 1843.34 seconds
