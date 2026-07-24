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
execution time: IAR + RelationalAnalysis = 7.82 + 243.95 = 251.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.3774763, upper bound: 0.3774810

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 340
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 289
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3505
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 301
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3195
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 2721
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2269
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2363

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3773313, upper bound: 0.3772184
time: 21.75 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3773311, upper bound: 0.3773423
time: 14.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 36.05 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 36.05
Output dim: 1, lower bound: -0.3773313, upper bound: 0.3772184
NS_A2, status: Status.UNKNOWN, split count: 1, time: 36.05
Output dim: 1, lower bound: -0.3773311, upper bound: 0.3773423

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2.1352377, -0.9083754, -2.1352603, -0.9080454, -0.6860197, 0.6859596
1: -0.5185103, 0.5046765, -0.5185230, 0.5047041, -0.8844474, 0.8844206
2: -2.0467234, -1.1159360, -2.0467892, -1.1159353, -0.4111045, 0.4111581
3: -1.7631890, -0.1258365, -1.7636502, -0.1258363, -0.9952543, 0.9954904
4: -2.5299592, -1.1365558, -2.5300765, -1.1365556, -0.4979886, 0.4981232
5: -1.7386382, -0.0888143, -1.7389393, -0.0888140, -1.0386860, 1.0388563
6: -2.4549615, -1.0445577, -2.4555271, -1.0445573, -0.5896482, 0.5902160
7: -1.9019297, -0.0775704, -1.9020272, -0.0775699, -1.1225172, 1.1225958
8: -1.4308974, -0.6562644, -1.4309058, -0.6562330, -0.3343658, 0.3343533
9: -0.3794448, 0.1251821, -0.3794568, 0.1251822, -0.2540144, 0.2540286

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 340
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 343
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 289
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 281
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 3505
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 3507
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 3261
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 288
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 301
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3195
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 2721
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 2722
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2269
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2723
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3177
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3419

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2346

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3770584, upper bound: 0.3771481
time: 39.64 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772376, upper bound: 0.3771422
time: 158.26 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.1424601, -0.9059468, -2.1354115, -0.9058279, -0.6967651, 0.6861123
1: -0.5223823, 0.5045388, -0.5185776, 0.5045652, -0.8893772, 0.8845959
2: -2.0464725, -1.1163299, -2.0465302, -1.1159313, -0.4112480, 0.4107901
3: -1.7647828, -0.1241969, -1.7650540, -0.1258361, -0.9942387, 0.9997431
4: -2.5294056, -1.1372539, -2.5295429, -1.1365540, -0.4983870, 0.5001702
5: -1.7385063, -0.0890019, -1.7389948, -0.0888131, -1.0368505, 1.0351245
6: -2.4592195, -1.0417527, -2.4593368, -1.0445554, -0.5913703, 0.5984164
7: -1.9002984, -0.0798488, -1.9006135, -0.0775645, -1.1206040, 1.1180217
8: -1.4306476, -0.6570138, -1.4309621, -0.6569035, -0.3364894, 0.3343722
9: -0.3797944, 0.1250527, -0.3795345, 0.1251836, -0.2547144, 0.2536215

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 340
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 343
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 289
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 281
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 3505
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 3507
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 2978
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2993
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 3261
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2563
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 3311
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 288
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 301
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3195
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 2721
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 2722
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2269
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2723
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2718
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3177
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2711
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3419

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2346

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3770604, upper bound: 0.3772377
time: 278.92 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772359, upper bound: 0.3772436
time: 17.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 302.82 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 302.82
Output dim: 1, lower bound: -0.3770584, upper bound: 0.3771481
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 302.82
Output dim: 1, lower bound: -0.3772376, upper bound: 0.3771422
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 302.82
Output dim: 1, lower bound: -0.3770604, upper bound: 0.3772377
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 302.82
Output dim: 1, lower bound: -0.3772359, upper bound: 0.3772436

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -2.1349547, -0.9118174, -2.1303172, -0.9117790, -0.6818868, 0.6772473
1: -0.5182986, 0.5041298, -0.5174066, 0.5040847, -0.8834555, 0.8816390
2: -2.0458131, -1.1159475, -2.0457718, -1.1152556, -0.4115798, 0.4104469
3: -1.7579758, -0.1258365, -1.7580559, -0.1239283, -0.9894537, 0.9888134
4: -2.5289135, -1.1365614, -2.5289283, -1.1356719, -0.4968657, 0.4963912
5: -1.7338951, -0.0888160, -1.7338405, -0.0861674, -1.0394489, 1.0349331
6: -2.4481778, -1.0445633, -2.4479225, -1.0466210, -0.5807706, 0.5826534
7: -1.9008783, -0.0775806, -1.9010082, -0.0738758, -1.1245849, 1.1212859
8: -1.4307930, -0.6567166, -1.4308223, -0.6567189, -0.3336626, 0.3329363
9: -0.3792897, 0.1251813, -0.3790598, 0.1254488, -0.2542154, 0.2534769

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 340
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 289
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3505
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 301
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3195
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 2721
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2269
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2389

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3770032, upper bound: 0.3769657
time: 69.63 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3770040, upper bound: 0.3770833
time: 14.44 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2.1352229, -0.9085417, -2.1352446, -0.9082310, -0.6803609, 0.6859468
1: -0.5184810, 0.5044002, -0.5184909, 0.5043954, -0.8834873, 0.8842227
2: -2.0462155, -1.1159375, -2.0462232, -1.1159368, -0.4105179, 0.4116292
3: -1.7617754, -0.1258364, -1.7621047, -0.1258365, -0.9945154, 0.9887831
4: -2.5293441, -1.1365579, -2.5293946, -1.1365579, -0.4975904, 0.4973268
5: -1.7365596, -0.0888146, -1.7366865, -0.0888142, -1.0359402, 1.0366637
6: -2.4545183, -1.0445584, -2.4550323, -1.0445583, -0.5892047, 0.5835601
7: -1.9002297, -0.0775712, -1.9001560, -0.0775705, -1.1209381, 1.1208744
8: -1.4308918, -0.6565511, -1.4308994, -0.6565531, -0.3334398, 0.3342028
9: -0.3794343, 0.1251817, -0.3794453, 0.1251819, -0.2538306, 0.2542866

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 340
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 289
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3505
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 301
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3195
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 2721
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2269
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2389

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3771787, upper bound: 0.3769745
time: 20.90 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3771757, upper bound: 0.3770862
time: 283.70 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2.1421790, -0.9093889, -2.1304698, -0.9095616, -0.6926316, 0.6774000
1: -0.5221725, 0.5039929, -0.5174603, 0.5039465, -0.8883850, 0.8818135
2: -2.0455639, -1.1163414, -2.0455143, -1.1152512, -0.4117244, 0.4100800
3: -1.7595696, -0.1241969, -1.7594603, -0.1239280, -0.9884385, 0.9930694
4: -2.5283613, -1.1372595, -2.5283976, -1.1356709, -0.4972624, 0.4984369
5: -1.7337633, -0.0890039, -1.7338960, -0.0861666, -1.0376168, 1.0312029
6: -2.4524343, -1.0417578, -2.4517310, -1.0466187, -0.5824918, 0.5908529
7: -1.8992496, -0.0798590, -1.8995976, -0.0738711, -1.1226735, 1.1167167
8: -1.4305441, -0.6574656, -1.4308770, -0.6573899, -0.3357869, 0.3329554
9: -0.3796386, 0.1250520, -0.3791386, 0.1254501, -0.2549158, 0.2530701

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 340
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 289
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3505
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 301
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3195
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 2721
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2269
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2389

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3770014, upper bound: 0.3770679
time: 56.10 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3770008, upper bound: 0.3770666
time: 180.36 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2.1424465, -0.9061129, -2.1353962, -0.9060138, -0.6911066, 0.6860994
1: -0.5223526, 0.5042626, -0.5185459, 0.5042564, -0.8884175, 0.8843974
2: -2.0459650, -1.1163311, -2.0459638, -1.1159329, -0.4106617, 0.4112606
3: -1.7633693, -0.1241969, -1.7635088, -0.1258361, -0.9935006, 0.9930367
4: -2.5287898, -1.1372563, -2.5288615, -1.1365565, -0.4979891, 0.4993736
5: -1.7364284, -0.0890024, -1.7367425, -0.0888135, -1.0341067, 1.0329330
6: -2.4587762, -1.0417535, -2.4588411, -1.0445564, -0.5909262, 0.5917600
7: -1.8986002, -0.0798499, -1.8987410, -0.0775653, -1.1190264, 1.1163015
8: -1.4306419, -0.6573002, -1.4309555, -0.6572239, -0.3355624, 0.3342219
9: -0.3797839, 0.1250525, -0.3795229, 0.1251832, -0.2545305, 0.2538795

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 340
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 289
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3505
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2978
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2993
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2563
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 3311
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 301
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3195
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 2721
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2269
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2718
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2711
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2389

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3771879, upper bound: 0.3770756
time: 460.02 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3771847, upper bound: 0.3771861
time: 359.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 825.77 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 825.77
Output dim: 1, lower bound: -0.3770032, upper bound: 0.3769657
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 825.77
Output dim: 1, lower bound: -0.3770040, upper bound: 0.3770833
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 825.77
Output dim: 1, lower bound: -0.3771787, upper bound: 0.3769745
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 825.77
Output dim: 1, lower bound: -0.3771757, upper bound: 0.3770862
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 825.77
Output dim: 1, lower bound: -0.3770014, upper bound: 0.3770679
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 825.77
Output dim: 1, lower bound: -0.3770008, upper bound: 0.3770666
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 825.77
Output dim: 1, lower bound: -0.3771879, upper bound: 0.3770756
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 825.77
Output dim: 1, lower bound: -0.3771847, upper bound: 0.3771861

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 251.77 + 2011.78 = 2263.55 seconds
