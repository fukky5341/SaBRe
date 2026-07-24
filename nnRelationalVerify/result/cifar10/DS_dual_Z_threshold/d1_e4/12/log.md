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
execution time: IAR + RelationalAnalysis = 7.85 + 134.18 = 142.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0304838, upper bound: 0.0304907

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3593

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3430

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304866, upper bound: 0.0304806
time: 97.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304776, upper bound: 0.0304853
time: 108.89 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 206.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 206.24
Output dim: 1, lower bound: -0.0304866, upper bound: 0.0304806
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 206.24
Output dim: 1, lower bound: -0.0304776, upper bound: 0.0304853

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2341066, 0.2341051
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1179477, 0.1179472
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6517189, 0.6517195
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9042087, 0.9042088
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4885550, 0.4885563
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0876980, 1.0876981
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5564325, 0.5564328
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.7065573, 0.7065575
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1878747, 0.1878746
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2545968, 0.2545969

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3593

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3441

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304843, upper bound: 0.0304567
time: 34.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304618, upper bound: 0.0304826
time: 92.26 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2341051, 0.2341066
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1179472, 0.1179477
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6517195, 0.6517189
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9042087, 0.9042087
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4885564, 0.4885551
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0876980, 1.0876980
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5564328, 0.5564325
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.7065575, 0.7065573
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1878746, 0.1878747
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2545969, 0.2545968

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3593

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3441

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304779, upper bound: 0.0304640
time: 159.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304538, upper bound: 0.0304897
time: 72.79 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 238.72 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 238.72
Output dim: 1, lower bound: -0.0304843, upper bound: 0.0304567
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 238.72
Output dim: 1, lower bound: -0.0304618, upper bound: 0.0304826
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 238.72
Output dim: 1, lower bound: -0.0304779, upper bound: 0.0304640
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 238.72
Output dim: 1, lower bound: -0.0304538, upper bound: 0.0304897

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2339594, 0.2339620
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1177161, 0.1177020
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6515093, 0.6515345
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9034265, 0.9033915
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4869560, 0.4870424
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0858768, 1.0857909
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5542833, 0.5543967
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.7034388, 0.7032787
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1859781, 0.1860802
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2544414, 0.2544326

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3593

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3440

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304845, upper bound: 0.0304388
time: 13.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304624, upper bound: 0.0304576
time: 90.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2339635, 0.2339579
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1177024, 0.1177157
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6515338, 0.6515100
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9033914, 0.9034266
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4870411, 0.4869573
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0857909, 1.0858768
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5543965, 0.5542835
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.7032785, 0.7034389
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1860803, 0.1859780
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2544325, 0.2544414

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3593

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3440

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304619, upper bound: 0.0304603
time: 27.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304380, upper bound: 0.0304854
time: 10.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2339579, 0.2339635
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1177157, 0.1177024
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6515100, 0.6515338
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9034266, 0.9033914
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4869573, 0.4870412
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0858768, 1.0857908
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5542835, 0.5543965
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.7034389, 0.7032786
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1859781, 0.1860803
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2544414, 0.2544325

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3593

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3440

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304757, upper bound: 0.0304439
time: 109.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304531, upper bound: 0.0304709
time: 12.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2339620, 0.2339594
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1177020, 0.1177161
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6515344, 0.6515093
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9033915, 0.9034265
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4870424, 0.4869560
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0857909, 1.0858767
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5543967, 0.5542833
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.7032787, 0.7034388
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1860802, 0.1859781
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2544326, 0.2544413

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3593

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3440

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304544, upper bound: 0.0304707
time: 14.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304307, upper bound: 0.0304916
time: 18.48 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 39.44 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 1, lower bound: -0.0304845, upper bound: 0.0304388
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 1, lower bound: -0.0304624, upper bound: 0.0304576
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 1, lower bound: -0.0304619, upper bound: 0.0304603
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 1, lower bound: -0.0304380, upper bound: 0.0304854
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 1, lower bound: -0.0304757, upper bound: 0.0304439
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 1, lower bound: -0.0304531, upper bound: 0.0304709
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 1, lower bound: -0.0304544, upper bound: 0.0304707
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 1, lower bound: -0.0304307, upper bound: 0.0304916

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2339590, 0.2339616
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1171719, 0.1171433
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6522557, 0.6523041
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9024420, 0.9023945
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4833576, 0.4835359
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0824076, 1.0822557
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5512523, 0.5514253
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.6976200, 0.6973153
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1821481, 0.1823462
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2541927, 0.2541775

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3593

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 263

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0304330, upper bound: 0.0304338
time: 226.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304835, upper bound: 0.0303844
time: 143.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2339590, 0.2339616
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1171575, 0.1171577
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6522790, 0.6522808
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9024296, 0.9024068
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4834493, 0.4834441
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0823417, 1.0823214
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5513118, 0.5513657
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.6974754, 0.6974592
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1822442, 0.1822501
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2541863, 0.2541838

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3593

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 263

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304122, upper bound: 0.0304596
time: 16.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304622, upper bound: 0.0304055
time: 302.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.0773776, -1.3378438, -2.0773776, -1.3378438, -0.2339632, 0.2339575
1: 0.2831002, 0.5404248, 0.2831002, 0.5404248, -0.1171582, 0.1171570
2: -2.4280314, -1.4616843, -2.4280314, -1.4616843, -0.6522802, 0.6522796
3: -3.0798190, -1.2350892, -3.0798190, -1.2350892, -0.9024068, 0.9024297
4: -4.1357274, -2.6985183, -4.1357274, -2.6985183, -0.4834428, 0.4834505
5: -3.2191141, -1.2183300, -3.2191141, -1.2183300, -1.0823213, 1.0823417
6: -2.8007994, -1.3527904, -2.8007994, -1.3527904, -0.5513655, 0.5513120
7: -4.5928125, -2.7272503, -4.5928125, -2.7272503, -0.6974590, 0.6974756
8: -0.9612234, -0.5790672, -0.9612234, -0.5790672, -0.1822502, 0.1822441
9: 0.0094709, 0.2804384, 0.0094709, 0.2804384, -0.2541838, 0.2541863

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 392
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3593

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 263

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0304106, upper bound: 0.0304337
time: 184.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0304612, upper bound: 0.0304051
time: 231.31 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 421.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 421.45
Output dim: 1, lower bound: -0.0304330, upper bound: 0.0304338
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 421.45
Output dim: 1, lower bound: -0.0304835, upper bound: 0.0303844
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 421.45
Output dim: 1, lower bound: -0.0304122, upper bound: 0.0304596
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 421.45
Output dim: 1, lower bound: -0.0304622, upper bound: 0.0304055
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 421.45
Output dim: 1, lower bound: -0.0304106, upper bound: 0.0304337
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 421.45
Output dim: 1, lower bound: -0.0304612, upper bound: 0.0304051
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 421.45
Output dim: 1, lower bound: -0.0304380, upper bound: 0.0304854
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 421.45
Output dim: 1, lower bound: -0.0304757, upper bound: 0.0304439
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 421.45
Output dim: 1, lower bound: -0.0304531, upper bound: 0.0304709
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 421.45
Output dim: 1, lower bound: -0.0304544, upper bound: 0.0304707
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 421.45
Output dim: 1, lower bound: -0.0304307, upper bound: 0.0304916

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 142.02 + 2021.51 = 2163.53 seconds
