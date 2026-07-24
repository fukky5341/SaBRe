## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 12)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0304594101


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2341630, 0.2341630)
1: (0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1179652, 0.1179652)
2: (-2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6516950, 0.6516950)
3: (-3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9041968, 0.9041968)
4: (-4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4885179, 0.4885179)
5: (-3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0876842, 1.0876842)
6: (-2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5563788, 0.5563788)
7: (-4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.7065411, 0.7065409)
8: (-0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1878704, 0.1878704)
9: (0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2545972, 0.2545973)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.89 + 137.69 = 145.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0304838, upper bound: 0.0304907

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2099

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304882, upper bound: 0.0304928
time: 36.81 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304882, upper bound: 0.0304892
time: 211.16 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 247.98 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 247.98
Output dim: 1, lower bound: -0.0304882, upper bound: 0.0304928
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 247.98
Output dim: 1, lower bound: -0.0304882, upper bound: 0.0304892

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2341630, 0.2341630
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1179652, 0.1179652
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6516950, 0.6516950
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9041968, 0.9041968
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4885179, 0.4885179
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0876842, 1.0876842
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5563788, 0.5563788
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.7065411, 0.7065409
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1878704, 0.1878704
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2545972, 0.2545973

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3087

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2235

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304858, upper bound: 0.0304900
time: 806.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304858, upper bound: 0.0304910
time: 112.68 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2341630, 0.2341630
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1179652, 0.1179652
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6516950, 0.6516950
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9041968, 0.9041968
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4885179, 0.4885179
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0876842, 1.0876842
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5563788, 0.5563788
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.7065411, 0.7065409
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1878704, 0.1878704
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2545972, 0.2545973

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2082

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304849, upper bound: 0.0304895
time: 101.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304861, upper bound: 0.0304899
time: 211.90 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 319.28 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 319.28
Output dim: 1, lower bound: -0.0304858, upper bound: 0.0304900
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 319.28
Output dim: 1, lower bound: -0.0304858, upper bound: 0.0304910
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 319.28
Output dim: 1, lower bound: -0.0304849, upper bound: 0.0304895
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 319.28
Output dim: 1, lower bound: -0.0304861, upper bound: 0.0304899

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2341630, 0.2341630
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1179652, 0.1179652
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6516950, 0.6516950
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9041968, 0.9041968
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4885179, 0.4885179
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0876842, 1.0876842
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5563788, 0.5563788
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.7065411, 0.7065409
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1878704, 0.1878704
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2545972, 0.2545973

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 452

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304866, upper bound: 0.0304886
time: 382.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304870, upper bound: 0.0304891
time: 261.29 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 649.82 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 649.82
Output dim: 1, lower bound: -0.0304866, upper bound: 0.0304886
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 649.82
Output dim: 1, lower bound: -0.0304870, upper bound: 0.0304891
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 649.82
Output dim: 1, lower bound: -0.0304858, upper bound: 0.0304910
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 649.82
Output dim: 1, lower bound: -0.0304849, upper bound: 0.0304895
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 649.82
Output dim: 1, lower bound: -0.0304861, upper bound: 0.0304899

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 145.58 + 2142.32 = 2287.91 seconds
