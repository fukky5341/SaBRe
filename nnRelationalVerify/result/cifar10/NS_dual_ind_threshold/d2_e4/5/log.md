## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 5)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.1215620163


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757330, 0.6757331)
1: (-3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906412, 2.8906415)
2: (0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283653, 0.2283653)
3: (-2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660600, 1.1660600)
4: (-2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448250, 0.8448251)
5: (-2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530630, 1.2530628)
6: (-6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485518, 0.9485518)
7: (-2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581674, 1.6581675)
8: (-2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877445, 2.5877442)
9: (-3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187610, 1.8187609)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 9.82 + 171.99 = 181.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.1216763, upper bound: 0.1216791

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 290
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 290

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216738, upper bound: 0.1213419
time: 157.37 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216676, upper bound: 0.1216919
time: 19.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 177.15 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 177.15
Output dim: 4, lower bound: -0.1216738, upper bound: 0.1213419
NS_A2, status: Status.UNKNOWN, split count: 1, time: 177.15
Output dim: 4, lower bound: -0.1216676, upper bound: 0.1216919

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.9373657, 1.8808306, 0.9349248, 1.8809463, -0.6716160, 0.6738549
1: -3.3615470, -0.0222447, -3.3617549, -0.0234571, -2.8871331, 2.8886862
2: 0.3615274, 1.0395275, 0.3618963, 1.0395817, -0.2259152, 0.2266775
3: -2.1354001, -0.3302180, -2.1357951, -0.3281451, -1.1618702, 1.1601797
4: -2.0165973, -0.6043231, -2.0227730, -0.6031386, -0.8281245, 0.8331597
5: -2.2459157, -0.3914326, -2.2465000, -0.3891162, -1.2472789, 1.2455174
6: -6.3506289, -3.0248570, -6.3520575, -3.0181332, -0.9335073, 0.9282063
7: -2.6405606, 0.3148725, -2.6426430, 0.3194125, -1.6489872, 1.6455460
8: -2.6678741, 0.1454751, -2.6741207, 0.1472137, -2.5671396, 2.5731621
9: -3.6063700, -1.2813771, -3.6068518, -1.2812908, -1.8158377, 1.8162994

Time for backsubstitution: 7.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3068

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216402, upper bound: 0.1212500
time: 188.42 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216351, upper bound: 0.1213184
time: 19.91 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.9346079, 1.8825628, 0.9346052, 1.8825642, -0.6755973, 0.6737180
1: -3.3612959, -0.0221312, -3.3615024, -0.0221319, -2.8885913, 2.8887255
2: 0.3617358, 1.0399849, 0.3617355, 1.0400127, -0.2283607, 0.2264537
3: -2.1392198, -0.3277416, -2.1392210, -0.3277415, -1.1647146, 1.1660478
4: -2.0234261, -0.5932908, -2.0234261, -0.5932891, -0.8448164, 0.8369397
5: -2.2512388, -0.3884901, -2.2512417, -0.3884903, -1.2505822, 1.2530471
6: -6.3655944, -3.0180764, -6.3656034, -3.0180767, -0.9275196, 0.9484583
7: -2.6508145, 0.3194544, -2.6508181, 0.3194544, -1.6476011, 1.6581103
8: -2.6749897, 0.1594672, -2.6749918, 0.1594716, -2.5888495, 2.5872324
9: -3.6082540, -1.2810321, -3.6083374, -1.2810183, -1.8183545, 1.8171256

Time for backsubstitution: 7.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3068

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216405, upper bound: 0.1216029
time: 278.81 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216360, upper bound: 0.1216576
time: 12.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 299.36 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 299.36
Output dim: 4, lower bound: -0.1216402, upper bound: 0.1212500
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 299.36
Output dim: 4, lower bound: -0.1216351, upper bound: 0.1213184
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 299.36
Output dim: 4, lower bound: -0.1216405, upper bound: 0.1216029
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 299.36
Output dim: 4, lower bound: -0.1216360, upper bound: 0.1216576

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.9374081, 1.8804915, 0.9349853, 1.8805692, -0.6712124, 0.6734558
1: -3.3610480, -0.0294120, -3.3591983, -0.0312476, -2.8790965, 2.8794067
2: 0.3623915, 1.0395266, 0.3628385, 1.0393944, -0.2249530, 0.2257921
3: -2.1289070, -0.3302199, -2.1287918, -0.3293453, -1.1538789, 1.1525995
4: -2.0133884, -0.6043354, -2.0192649, -0.6030324, -0.8244765, 0.8295637
5: -2.2388167, -0.3914403, -2.2388010, -0.3909526, -1.2385876, 1.2374589
6: -6.3444805, -3.0248594, -6.3453145, -3.0182426, -0.9257176, 0.9209112
7: -2.6348171, 0.3148610, -2.6362858, 0.3172617, -1.6414526, 1.6392047
8: -2.6677113, 0.1423702, -2.6732473, 0.1438048, -2.5636640, 2.5693033
9: -3.6060817, -1.2830882, -3.6058831, -1.2831750, -1.8134125, 1.8128386

Time for backsubstitution: 7.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215381, upper bound: 0.1211868
time: 213.83 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216067, upper bound: 0.1211896
time: 333.54 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.9373795, 1.8806596, 0.9349391, 1.8807603, -0.6714061, 0.6737127
1: -3.3614399, -0.0233085, -3.3616433, -0.0245357, -2.8846173, 2.8875473
2: 0.3616253, 1.0395274, 0.3619989, 1.0395815, -0.2258441, 0.2258631
3: -2.1340237, -0.3302192, -2.1343091, -0.3281462, -1.1612337, 1.1534208
4: -2.0151644, -0.6043252, -2.0212123, -0.6031410, -0.8271327, 0.8299626
5: -2.2445545, -0.3914360, -2.2450304, -0.3891204, -1.2466981, 1.2383361
6: -6.3493142, -3.0248590, -6.3506622, -3.0181339, -0.9325356, 0.9218501
7: -2.6393793, 0.3148693, -2.6413741, 0.3194090, -1.6486945, 1.6381801
8: -2.6678195, 0.1444928, -2.6740627, 0.1461648, -2.5658908, 2.5720577
9: -3.6062846, -1.2821225, -3.6067619, -1.2821120, -1.8151381, 1.8156912

Time for backsubstitution: 7.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215421, upper bound: 0.1212758
time: 449.87 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216033, upper bound: 0.1212868
time: 14.50 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.9346501, 1.8822230, 0.9346659, 1.8821872, -0.6751940, 0.6733190
1: -3.3607824, -0.0293014, -3.3589354, -0.0299227, -2.8805468, 2.8794413
2: 0.3625994, 1.0399842, 0.3626776, 1.0398256, -0.2273985, 0.2255680
3: -2.1327262, -0.3277431, -2.1322176, -0.3289408, -1.1567215, 1.1584675
4: -2.0202100, -0.5933040, -2.0199156, -0.5931832, -0.8411607, 0.8333380
5: -2.2441394, -0.3884984, -2.2435417, -0.3903269, -1.2418894, 1.2449887
6: -6.3594475, -3.0180793, -6.3588619, -3.0181863, -0.9197301, 0.9411635
7: -2.6450715, 0.3194431, -2.6444609, 0.3173044, -1.6400647, 1.6517639
8: -2.6748269, 0.1563623, -2.6741176, 0.1560626, -2.5853734, 2.5833719
9: -3.6079628, -1.2827437, -3.6073663, -1.2829027, -1.8159274, 1.8136626

Time for backsubstitution: 7.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215373, upper bound: 0.1215454
time: 15.82 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216065, upper bound: 0.1215333
time: 375.09 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.9346215, 1.8823916, 0.9346199, 1.8823781, -0.6753876, 0.6735759
1: -3.3611860, -0.0231941, -3.3613887, -0.0232081, -2.8860760, 2.8875873
2: 0.3618335, 1.0399847, 0.3618379, 1.0400127, -0.2282899, 0.2256388
3: -2.1378431, -0.3277426, -2.1377344, -0.3277424, -1.1640780, 1.1592875
4: -2.0219893, -0.5932928, -2.0218642, -0.5932914, -0.8438225, 0.8337383
5: -2.2498775, -0.3884941, -2.2497716, -0.3884943, -1.2500014, 1.2458642
6: -6.3642793, -3.0180779, -6.3642092, -3.0180783, -0.9265479, 0.9421021
7: -2.6496339, 0.3194519, -2.6495495, 0.3194521, -1.6473079, 1.6507404
8: -2.6749353, 0.1584836, -2.6749346, 0.1584224, -2.5876002, 2.5861273
9: -3.6081684, -1.2817781, -3.6082468, -1.2818396, -1.8176541, 1.8165164

Time for backsubstitution: 7.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215452, upper bound: 0.1216195
time: 36.18 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216065, upper bound: 0.1216224
time: 20.00 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 63.89 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 63.89
Output dim: 4, lower bound: -0.1215381, upper bound: 0.1211868
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 63.89
Output dim: 4, lower bound: -0.1216067, upper bound: 0.1211896
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 63.89
Output dim: 4, lower bound: -0.1215421, upper bound: 0.1212758
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 63.89
Output dim: 4, lower bound: -0.1216033, upper bound: 0.1212868
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 63.89
Output dim: 4, lower bound: -0.1215373, upper bound: 0.1215454
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 63.89
Output dim: 4, lower bound: -0.1216065, upper bound: 0.1215333
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 63.89
Output dim: 4, lower bound: -0.1215452, upper bound: 0.1216195
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 63.89
Output dim: 4, lower bound: -0.1216065, upper bound: 0.1216224

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9372837, 1.8801252, 0.9350208, 1.8802338, -0.6710379, 0.6730476
1: -3.3686459, -0.0343111, -3.3588545, -0.0359530, -2.8828578, 2.8741405
2: 0.3628681, 1.0399134, 0.3633456, 1.0393938, -0.2243294, 0.2261914
3: -2.1230791, -0.3265370, -2.1233447, -0.3293461, -1.1481743, 1.1564925
4: -2.0110252, -0.6029208, -2.0170608, -0.6030406, -0.8221990, 0.8307160
5: -2.2330828, -0.3879315, -2.2333574, -0.3909580, -1.2326670, 1.2408804
6: -6.3409925, -3.0217621, -6.3420081, -3.0182433, -0.9204854, 0.9235470
7: -2.6299555, 0.3187482, -2.6318095, 0.3172541, -1.6367775, 1.6439412
8: -2.6690385, 0.1394637, -2.6731067, 0.1411397, -2.5623653, 2.5663440
9: -3.6083202, -1.2847824, -3.6056423, -1.2847294, -1.8139740, 1.8109992

Time for backsubstitution: 7.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215669, upper bound: 0.1210446
time: 21.60 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215721, upper bound: 0.1210380
time: 154.49 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9372437, 1.8803847, 0.9349644, 1.8805115, -0.6713180, 0.6734053
1: -3.3692036, -0.0255947, -3.3614528, -0.0268095, -2.8908067, 2.8848109
2: 0.3618255, 1.0399144, 0.3622425, 1.0395812, -0.2252575, 0.2262081
3: -2.1307330, -0.3265361, -2.1312160, -0.3281471, -1.1554513, 1.1565957
4: -2.0139060, -0.6029063, -2.0200193, -0.6031449, -0.8249798, 0.8309319
5: -2.2413054, -0.3879259, -2.2418880, -0.3891244, -1.2408476, 1.2411944
6: -6.3481832, -3.0217605, -6.3495364, -3.0181353, -0.9276234, 0.9244981
7: -2.6364484, 0.3187602, -2.6386781, 0.3194045, -1.6432220, 1.6412628
8: -2.6691897, 0.1424195, -2.6739607, 0.1442532, -2.5652971, 2.5698972
9: -3.6086054, -1.2832174, -3.6065962, -1.2831159, -1.8161703, 1.8143768

Time for backsubstitution: 7.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215703, upper bound: 0.1211209
time: 290.13 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215754, upper bound: 0.1212439
time: 286.77 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9345252, 1.8818570, 0.9347015, 1.8818514, -0.6750203, 0.6729112
1: -3.3683677, -0.0342016, -3.3585856, -0.0346277, -2.8843026, 2.8741710
2: 0.3630756, 1.0403711, 0.3631845, 1.0398251, -0.2267751, 0.2259670
3: -2.1268983, -0.3240604, -2.1267703, -0.3289417, -1.1510170, 1.1623602
4: -2.0178432, -0.5918895, -2.0177095, -0.5931921, -0.8388762, 0.8344874
5: -2.2384052, -0.3849891, -2.2380981, -0.3903324, -1.2359681, 1.2484095
6: -6.3559594, -3.0149810, -6.3555551, -3.0181878, -0.9144977, 0.9437997
7: -2.6402092, 0.3233306, -2.6399841, 0.3172970, -1.6353898, 1.6564999
8: -2.6761568, 0.1534551, -2.6739776, 0.1533968, -2.5840764, 2.5804133
9: -3.6101985, -1.2844380, -3.6071246, -1.2844570, -1.8164872, 1.8118219

Time for backsubstitution: 7.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215691, upper bound: 0.1213831
time: 25.16 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215799, upper bound: 0.1215120
time: 15.63 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9346601, 1.8820311, 0.9346555, 1.8820527, -0.6750181, 0.6731628
1: -3.3607135, -0.0299754, -3.3609524, -0.0294487, -2.8794973, 2.8803670
2: 0.3625462, 1.0399841, 0.3624904, 1.0400121, -0.2274008, 0.2249480
3: -2.1310003, -0.3277434, -2.1314366, -0.3277436, -1.1564095, 1.1537774
4: -2.0189013, -0.5933056, -2.0190556, -0.5933031, -0.8404706, 0.8311777
5: -2.2432015, -0.3885000, -2.2436068, -0.3884998, -1.2420448, 1.2400552
6: -6.3587613, -3.0180793, -6.3591037, -3.0180793, -0.9195038, 0.9366662
7: -2.6442747, 0.3194415, -2.6445744, 0.3194422, -1.6408834, 1.6464322
8: -2.6747875, 0.1555914, -2.6747975, 0.1557559, -2.5849185, 2.5832112
9: -3.6079021, -1.2837064, -3.6080012, -1.2835805, -1.8159666, 1.8146877

Time for backsubstitution: 7.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215033, upper bound: 0.1214510
time: 239.71 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215089, upper bound: 0.1215881
time: 257.15 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 181.81 + 3504.67 = 3686.48 seconds
