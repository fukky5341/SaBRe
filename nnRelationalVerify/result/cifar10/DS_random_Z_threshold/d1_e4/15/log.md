## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 15)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.3770988237


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.1354892, -0.9046922, -2.1354892, -0.9046922, -0.6896927, 0.6896927)
1: (-0.5186762, 0.5049943, -0.5186762, 0.5049943, -0.8851292, 0.8851292)
2: (-2.0474486, -1.1159297, -2.0474486, -1.1159297, -0.4118208, 0.4118208)
3: (-1.7686293, -0.1258358, -1.7686293, -0.1258358, -0.9997908, 0.9997908)
4: (-2.5312419, -1.1365529, -2.5312419, -1.1365529, -0.5000231, 0.5000231)
5: (-1.7435472, -0.0888126, -1.7435472, -0.0888126, -1.0413697, 1.0413697)
6: (-2.4612699, -1.0445554, -2.4612699, -1.0445554, -0.5961789, 0.5961789)
7: (-1.9032533, -0.0775610, -1.9032533, -0.0775610, -1.1237259, 1.1237259)
8: (-1.4309907, -0.6558606, -1.4309907, -0.6558606, -0.3350431, 0.3350431)
9: (-0.3795788, 0.1251844, -0.3795788, 0.1251844, -0.2541911, 0.2541912)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.86 + 253.35 = 261.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.3774763, upper bound: 0.3774810

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 281
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2722
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2430

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3104

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3774751, upper bound: 0.3774839
time: 22.64 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3774751, upper bound: 0.3774795
time: 25.13 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 47.78 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 47.78
Output dim: 1, lower bound: -0.3774751, upper bound: 0.3774839
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 47.78
Output dim: 1, lower bound: -0.3774751, upper bound: 0.3774795

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.1354892, -0.9046922, -2.1354892, -0.9046922, -0.6896927, 0.6896927
1: -0.5186762, 0.5049943, -0.5186762, 0.5049943, -0.8851292, 0.8851292
2: -2.0474486, -1.1159297, -2.0474486, -1.1159297, -0.4118208, 0.4118208
3: -1.7686293, -0.1258358, -1.7686293, -0.1258358, -0.9997908, 0.9997908
4: -2.5312419, -1.1365529, -2.5312419, -1.1365529, -0.5000231, 0.5000231
5: -1.7435472, -0.0888126, -1.7435472, -0.0888126, -1.0413697, 1.0413697
6: -2.4612699, -1.0445554, -2.4612699, -1.0445554, -0.5961789, 0.5961789
7: -1.9032533, -0.0775610, -1.9032533, -0.0775610, -1.1237259, 1.1237259
8: -1.4309907, -0.6558606, -1.4309907, -0.6558606, -0.3350431, 0.3350431
9: -0.3795788, 0.1251844, -0.3795788, 0.1251844, -0.2541911, 0.2541912

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 281
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2722
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 3177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3507

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3774130, upper bound: 0.3774752
time: 933.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3774769, upper bound: 0.3774247
time: 351.11 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.1354892, -0.9046922, -2.1354892, -0.9046922, -0.6896927, 0.6896927
1: -0.5186762, 0.5049943, -0.5186762, 0.5049943, -0.8851292, 0.8851292
2: -2.0474486, -1.1159297, -2.0474486, -1.1159297, -0.4118208, 0.4118208
3: -1.7686293, -0.1258358, -1.7686293, -0.1258358, -0.9997908, 0.9997908
4: -2.5312419, -1.1365529, -2.5312419, -1.1365529, -0.5000231, 0.5000231
5: -1.7435472, -0.0888126, -1.7435472, -0.0888126, -1.0413697, 1.0413697
6: -2.4612699, -1.0445554, -2.4612699, -1.0445554, -0.5961789, 0.5961789
7: -1.9032533, -0.0775610, -1.9032533, -0.0775610, -1.1237259, 1.1237259
8: -1.4309907, -0.6558606, -1.4309907, -0.6558606, -0.3350431, 0.3350431
9: -0.3795788, 0.1251844, -0.3795788, 0.1251844, -0.2541911, 0.2541912

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 2711
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2721
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2722
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2440
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 281
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2269
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 289
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 3044

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 352

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3774781, upper bound: 0.3771528
time: 172.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3771487, upper bound: 0.3774775
time: 184.64 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 363.74 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 363.74
Output dim: 1, lower bound: -0.3774130, upper bound: 0.3774752
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 363.74
Output dim: 1, lower bound: -0.3774769, upper bound: 0.3774247
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 363.74
Output dim: 1, lower bound: -0.3774781, upper bound: 0.3771528
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 363.74
Output dim: 1, lower bound: -0.3771487, upper bound: 0.3774775

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 261.22 + 1702.24 = 1963.46 seconds
