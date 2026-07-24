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
execution time: IAR + RelationalAnalysis = 7.71 + 161.04 = 168.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.1216763, upper bound: 0.1216791

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 500

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2588

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216374, upper bound: 0.1215738
time: 159.41 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215759, upper bound: 0.1216576
time: 136.26 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 295.69 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 295.69
Output dim: 4, lower bound: -0.1216374, upper bound: 0.1215738
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 295.69
Output dim: 4, lower bound: -0.1215759, upper bound: 0.1216576

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6756870, 0.6756868
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8900180, 2.8900435
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2275521, 0.2275307
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1617850, 1.1616945
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8434529, 0.8434105
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2486151, 1.2485195
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9413029, 0.9412856
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6533456, 1.6532155
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5876939, 2.5876956
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8185741, 1.8185744

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2059

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2148

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214644, upper bound: 0.1215880
time: 86.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216345, upper bound: 0.1214097
time: 31.21 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6756867, 0.6756870
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8900433, 2.8900177
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2275307, 0.2275521
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1616944, 1.1617849
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8434105, 0.8434530
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2485195, 1.2486153
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9412855, 0.9413030
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6532154, 1.6533457
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5876958, 2.5876937
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8185745, 1.8185740

Time for backsubstitution: 5.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3027

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2060

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215191, upper bound: 0.1215877
time: 96.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215140, upper bound: 0.1215937
time: 161.44 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 263.69 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 263.69
Output dim: 4, lower bound: -0.1214644, upper bound: 0.1215880
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 263.69
Output dim: 4, lower bound: -0.1216345, upper bound: 0.1214097
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 263.69
Output dim: 4, lower bound: -0.1215191, upper bound: 0.1215877
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 263.69
Output dim: 4, lower bound: -0.1215140, upper bound: 0.1215937

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6755730, 0.6755715
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8902776, 2.8903079
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2269003, 0.2268716
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1542916, 1.1543630
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8416860, 0.8416833
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2408268, 1.2408987
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9398663, 0.9398472
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6426831, 1.6427648
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877347, 2.5877371
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8186117, 1.8186123

Time for backsubstitution: 5.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2903

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3011

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214598, upper bound: 0.1215784
time: 144.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214597, upper bound: 0.1215922
time: 14.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6755717, 0.6755728
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8902829, 2.8903031
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2268930, 0.2268789
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1544535, 1.1542013
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8417258, 0.8416433
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2409945, 1.2407312
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9398646, 0.9398488
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6428949, 1.6425529
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877352, 2.5877366
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8186117, 1.8186123

Time for backsubstitution: 5.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 889

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 67

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216173, upper bound: 0.1213406
time: 222.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215794, upper bound: 0.1214003
time: 18.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6756817, 0.6756747
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8900013, 2.8900204
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2275297, 0.2275514
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1616548, 1.1617653
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8433967, 0.8434395
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2484853, 1.2485987
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9412550, 0.9412832
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6532009, 1.6533356
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5876939, 2.5876923
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8185700, 1.8185704

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 434

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 79

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1214080, upper bound: 0.1214804
time: 16.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1214049, upper bound: 0.1214815
time: 24.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6756746, 0.6756819
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8900452, 2.8899765
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2275300, 0.2275511
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1616749, 1.1617454
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8433967, 0.8434393
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2485030, 1.2485811
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9412657, 0.9412725
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6532056, 1.6533312
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5876944, 2.5876918
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8185710, 1.8185697

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2988

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2059

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215073, upper bound: 0.1215887
time: 153.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215010, upper bound: 0.1215941
time: 127.00 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 286.16 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 286.16
Output dim: 4, lower bound: -0.1214598, upper bound: 0.1215784
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 286.16
Output dim: 4, lower bound: -0.1214597, upper bound: 0.1215922
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 286.16
Output dim: 4, lower bound: -0.1216173, upper bound: 0.1213406
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 286.16
Output dim: 4, lower bound: -0.1215794, upper bound: 0.1214003
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 286.16
Output dim: 4, lower bound: -0.1214080, upper bound: 0.1214804
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 286.16
Output dim: 4, lower bound: -0.1214049, upper bound: 0.1214815
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 286.16
Output dim: 4, lower bound: -0.1215073, upper bound: 0.1215887
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 286.16
Output dim: 4, lower bound: -0.1215010, upper bound: 0.1215941

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6753619, 0.6753533
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8903823, 2.8904245
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2266098, 0.2265803
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1525099, 1.1525810
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8416181, 0.8416140
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2394686, 1.2395399
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9349476, 0.9349488
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6417559, 1.6418331
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5876920, 2.5876946
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8184631, 1.8184663

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2885

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2891

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214597, upper bound: 0.1215846
time: 157.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214586, upper bound: 0.1215796
time: 151.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6753548, 0.6753604
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8903942, 2.8904126
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2266090, 0.2265811
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1525097, 1.1525812
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8416169, 0.8416153
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2394681, 1.2395403
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9349679, 0.9349285
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6417512, 1.6418378
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5876925, 2.5876942
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8184657, 1.8184637

Time for backsubstitution: 6.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2095

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1214465, upper bound: 0.1215596
time: 12.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214329, upper bound: 0.1215661
time: 96.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6755029, 0.6755004
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8902752, 2.8903022
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2267958, 0.2267897
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1537290, 1.1535234
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8417199, 0.8416330
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2402503, 1.2400330
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9387400, 0.9387691
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6428570, 1.6425176
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877271, 2.5877299
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8185734, 1.8185742

Time for backsubstitution: 6.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2893

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216134, upper bound: 0.1213473
time: 336.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216185, upper bound: 0.1213512
time: 27.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6754993, 0.6755040
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8902819, 2.8902955
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2268039, 0.2267817
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1537757, 1.1534767
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8417156, 0.8416375
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2402961, 1.2399871
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9387848, 0.9387243
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6428598, 1.6425149
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877290, 2.5877285
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8185736, 1.8185737

Time for backsubstitution: 6.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2903

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215631, upper bound: 0.1213976
time: 17.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215635, upper bound: 0.1213921
time: 24.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6756405, 0.6756388
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906295, 2.8906174
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2275259, 0.2275469
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1609043, 1.1609957
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8430076, 0.8430499
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2480919, 1.2481787
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9406467, 0.9406682
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6532785, 1.6534050
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877066, 2.5877042
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187157, 1.8187146

Time for backsubstitution: 6.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2652

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 835

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214893, upper bound: 0.1215706
time: 35.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215043, upper bound: 0.1215724
time: 24.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6756312, 0.6756485
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8906896, 2.8905602
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2275258, 0.2275471
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1609257, 1.1609749
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8430073, 0.8430502
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2481006, 1.2481699
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9406624, 0.9406534
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6532818, 1.6534042
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877066, 2.5877037
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8187159, 1.8187144

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2963

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 354

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1212970, upper bound: 0.1216040
time: 22.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1214997, upper bound: 0.1214472
time: 22.54 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 50.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1214597, upper bound: 0.1215846
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1214586, upper bound: 0.1215796
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1214465, upper bound: 0.1215596
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1214329, upper bound: 0.1215661
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1216134, upper bound: 0.1213473
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1216185, upper bound: 0.1213512
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1215631, upper bound: 0.1213976
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1215635, upper bound: 0.1213921
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1214893, upper bound: 0.1215706
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1215043, upper bound: 0.1215724
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1212970, upper bound: 0.1216040
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 50.96
Output dim: 4, lower bound: -0.1214997, upper bound: 0.1214472

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6753592, 0.6753506
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8903809, 2.8904235
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2266037, 0.2265741
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1525084, 1.1525795
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8416169, 0.8416117
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2394663, 1.2395377
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9349445, 0.9349458
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6417536, 1.6418304
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5876868, 2.5876892
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8184576, 1.8184608

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2885

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214586, upper bound: 0.1215803
time: 16.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214644, upper bound: 0.1215817
time: 376.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6753591, 0.6753507
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8903809, 2.8904235
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2266037, 0.2265742
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1525084, 1.1525795
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8416159, 0.8416127
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2394663, 1.2395376
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9349445, 0.9349458
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6417536, 1.6418304
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5876873, 2.5876892
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8184576, 1.8184608

Time for backsubstitution: 6.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 888

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 890

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214469, upper bound: 0.1215621
time: 28.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214469, upper bound: 0.1215665
time: 294.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6750054, 0.6750188
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8909695, 2.8909698
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2262878, 0.2262573
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1488941, 1.1488717
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8415391, 0.8415448
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2356758, 1.2356499
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9284725, 0.9282733
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6403848, 1.6403769
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877321, 2.5877342
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8188152, 1.8188020

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3438

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2337

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214303, upper bound: 0.1215773
time: 11.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214302, upper bound: 0.1215696
time: 139.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6755030, 0.6755005
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8902752, 2.8903022
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2267958, 0.2267897
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1537287, 1.1535233
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8417199, 0.8416330
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2402502, 1.2400329
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9387398, 0.9387689
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6428567, 1.6425173
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877271, 2.5877302
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8185736, 1.8185742

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2346

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 835

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215988, upper bound: 0.1213500
time: 18.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216086, upper bound: 0.1213219
time: 19.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9345793, 1.8825785, 0.9345793, 1.8825785, -0.6755030, 0.6755005
1: -3.3633168, -0.0221295, -3.3633168, -0.0221295, -2.8902752, 2.8903022
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2267958, 0.2267897
3: -2.1392314, -0.3277394, -2.1392314, -0.3277394, -1.1537287, 1.1535233
4: -2.0234282, -0.5932750, -2.0234282, -0.5932750, -0.8417199, 0.8416330
5: -2.2512624, -0.3884892, -2.2512624, -0.3884892, -1.2402502, 1.2400329
6: -6.3656793, -3.0180764, -6.3656793, -3.0180764, -0.9387398, 0.9387689
7: -2.6508508, 0.3194576, -2.6508508, 0.3194576, -1.6428567, 1.6425173
8: -2.6750133, 0.1594996, -2.6750133, 0.1594996, -2.5877271, 2.5877302
9: -3.6092691, -1.2809148, -3.6092691, -1.2809148, -1.8185736, 1.8185742

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 2890
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3516
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2922

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215873, upper bound: 0.1213340
time: 214.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215931, upper bound: 0.1213181
time: 304.94 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 525.94 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 525.94
Output dim: 4, lower bound: -0.1214586, upper bound: 0.1215803
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 525.94
Output dim: 4, lower bound: -0.1214644, upper bound: 0.1215817
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 525.94
Output dim: 4, lower bound: -0.1214469, upper bound: 0.1215621
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 525.94
Output dim: 4, lower bound: -0.1214469, upper bound: 0.1215665
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 525.94
Output dim: 4, lower bound: -0.1214303, upper bound: 0.1215773
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 525.94
Output dim: 4, lower bound: -0.1214302, upper bound: 0.1215696
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 525.94
Output dim: 4, lower bound: -0.1215988, upper bound: 0.1213500
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 525.94
Output dim: 4, lower bound: -0.1216086, upper bound: 0.1213219
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 525.94
Output dim: 4, lower bound: -0.1215873, upper bound: 0.1213340
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 525.94
Output dim: 4, lower bound: -0.1215931, upper bound: 0.1213181
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 525.94
Output dim: 4, lower bound: -0.1215631, upper bound: 0.1213976
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 525.94
Output dim: 4, lower bound: -0.1215635, upper bound: 0.1213921
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 525.94
Output dim: 4, lower bound: -0.1214893, upper bound: 0.1215706
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 525.94
Output dim: 4, lower bound: -0.1215043, upper bound: 0.1215724
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 525.94
Output dim: 4, lower bound: -0.1212970, upper bound: 0.1216040

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 168.75 + 3850.22 = 4018.97 seconds
