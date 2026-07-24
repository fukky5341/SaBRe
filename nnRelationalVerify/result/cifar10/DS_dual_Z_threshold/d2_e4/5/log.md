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
execution time: IAR + RelationalAnalysis = 9.61 + 170.68 = 180.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.1216763, upper bound: 0.1216791

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 3129

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216557, upper bound: 0.1216902
time: 11.12 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216685, upper bound: 0.1216597
time: 264.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 275.85 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 275.85
Output dim: 4, lower bound: -0.1216557, upper bound: 0.1216902
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 275.85
Output dim: 4, lower bound: -0.1216685, upper bound: 0.1216597

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757292, 0.6757292
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906412, 2.8906412
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283646, 0.2283646
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660600, 1.1660600
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448230, 0.8448230
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530627, 1.2530627
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485496, 0.9485497
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581643, 1.6581644
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877402, 2.5877402
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187605, 1.8187604

Time for backsubstitution: 7.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 2439

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216226, upper bound: 0.1216695
time: 18.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216467, upper bound: 0.1216569
time: 170.59 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757292, 0.6757292
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906412, 2.8906412
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283646, 0.2283646
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660600, 1.1660600
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448230, 0.8448230
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530627, 1.2530627
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485499, 0.9485497
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581643, 1.6581644
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877402, 2.5877402
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187605, 1.8187604

Time for backsubstitution: 7.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 2439

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216483, upper bound: 0.1216624
time: 12.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216568, upper bound: 0.1216503
time: 10.89 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.29 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 4, lower bound: -0.1216226, upper bound: 0.1216695
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 4, lower bound: -0.1216467, upper bound: 0.1216569
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 4, lower bound: -0.1216483, upper bound: 0.1216624
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 4, lower bound: -0.1216568, upper bound: 0.1216503

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757284, 0.6757284
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906412, 2.8906412
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283638, 0.2283638
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660597, 1.1660597
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448220, 0.8448222
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530622, 1.2530622
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485495, 0.9485496
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581633, 1.6581632
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877402, 2.5877402
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187598, 1.8187599

Time for backsubstitution: 7.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 3109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216052, upper bound: 0.1216578
time: 29.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216230, upper bound: 0.1216341
time: 244.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757284, 0.6757284
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906412, 2.8906412
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283638, 0.2283638
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660597, 1.1660597
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448220, 0.8448222
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530622, 1.2530622
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485495, 0.9485496
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581631, 1.6581633
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877402, 2.5877402
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187600, 1.8187597

Time for backsubstitution: 7.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 3109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216182, upper bound: 0.1216472
time: 167.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216334, upper bound: 0.1216234
time: 291.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757284, 0.6757284
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906412, 2.8906412
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283638, 0.2283638
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660597, 1.1660597
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448220, 0.8448222
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530622, 1.2530622
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485495, 0.9485496
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581633, 1.6581632
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877402, 2.5877402
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187598, 1.8187599

Time for backsubstitution: 7.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 3109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216169, upper bound: 0.1216412
time: 163.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216330, upper bound: 0.1216260
time: 132.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757284, 0.6757284
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906412, 2.8906412
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283638, 0.2283638
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660597, 1.1660597
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448220, 0.8448222
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530622, 1.2530622
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485495, 0.9485495
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581631, 1.6581633
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877402, 2.5877402
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187600, 1.8187597

Time for backsubstitution: 7.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 3109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216297, upper bound: 0.1216307
time: 120.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216428, upper bound: 0.1216147
time: 214.87 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 342.96 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 342.96
Output dim: 4, lower bound: -0.1216052, upper bound: 0.1216578
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 342.96
Output dim: 4, lower bound: -0.1216230, upper bound: 0.1216341
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 342.96
Output dim: 4, lower bound: -0.1216182, upper bound: 0.1216472
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 342.96
Output dim: 4, lower bound: -0.1216334, upper bound: 0.1216234
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 342.96
Output dim: 4, lower bound: -0.1216169, upper bound: 0.1216412
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 342.96
Output dim: 4, lower bound: -0.1216330, upper bound: 0.1216260
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 342.96
Output dim: 4, lower bound: -0.1216297, upper bound: 0.1216307
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 342.96
Output dim: 4, lower bound: -0.1216428, upper bound: 0.1216147

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757284, 0.6757285
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906388, 2.8906388
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660595, 1.1660595
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448218, 0.8448218
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530622, 1.2530622
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485492, 0.9485493
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581619, 1.6581619
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877397, 2.5877399
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187568, 1.8187568

Time for backsubstitution: 7.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 2646

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215358, upper bound: 0.1215893
time: 203.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215322, upper bound: 0.1215824
time: 322.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757284, 0.6757285
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906388, 2.8906393
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660595, 1.1660596
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448218, 0.8448218
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530622, 1.2530622
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485492, 0.9485493
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581619, 1.6581619
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877397, 2.5877399
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187568, 1.8187571

Time for backsubstitution: 7.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 2646

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215632, upper bound: 0.1215766
time: 16.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215560, upper bound: 0.1215587
time: 241.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757284, 0.6757285
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906388, 2.8906388
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660595, 1.1660595
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448218, 0.8448218
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530622, 1.2530622
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485492, 0.9485492
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581619, 1.6581620
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877397, 2.5877399
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187568, 1.8187568

Time for backsubstitution: 7.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 2646

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215417, upper bound: 0.1215768
time: 19.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215424, upper bound: 0.1215782
time: 26.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757284, 0.6757285
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906388, 2.8906393
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660595, 1.1660595
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448218, 0.8448218
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530622, 1.2530622
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485492, 0.9485493
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581619, 1.6581619
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877397, 2.5877399
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187568, 1.8187568

Time for backsubstitution: 7.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 2646

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215667, upper bound: 0.1215675
time: 11.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215642, upper bound: 0.1215578
time: 317.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757284, 0.6757285
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906388, 2.8906388
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660595, 1.1660595
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448218, 0.8448218
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530622, 1.2530622
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485492, 0.9485493
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581619, 1.6581619
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877397, 2.5877399
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187568, 1.8187568

Time for backsubstitution: 7.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 2646

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215471, upper bound: 0.1215746
time: 85.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215533, upper bound: 0.1215753
time: 48.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6757284, 0.6757285
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906388, 2.8906393
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1660595, 1.1660596
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8448218, 0.8448218
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2530622, 1.2530622
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9485492, 0.9485493
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6581619, 1.6581619
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877397, 2.5877399
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187568, 1.8187568

Time for backsubstitution: 7.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 2646

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215722, upper bound: 0.1215584
time: 145.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215633, upper bound: 0.1215512
time: 184.18 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 337.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215358, upper bound: 0.1215893
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215322, upper bound: 0.1215824
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215632, upper bound: 0.1215766
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215560, upper bound: 0.1215587
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215417, upper bound: 0.1215768
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215424, upper bound: 0.1215782
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215667, upper bound: 0.1215675
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215642, upper bound: 0.1215578
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215471, upper bound: 0.1215746
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215533, upper bound: 0.1215753
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215722, upper bound: 0.1215584
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 337.41
Output dim: 4, lower bound: -0.1215633, upper bound: 0.1215512
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 337.41
Output dim: 4, lower bound: -0.1216297, upper bound: 0.1216307
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 337.41
Output dim: 4, lower bound: -0.1216428, upper bound: 0.1216147

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 180.29 + 3569.98 = 3750.28 seconds
