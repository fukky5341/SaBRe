## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 7)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0673145181


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2783225, 0.2783225)
1: (0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0524148, 0.0524148)
2: (-1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5569975, 0.5569975)
3: (-2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5425195, 0.5425195)
4: (-1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7700787, 0.7700785)
5: (-3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3982390, 0.3982390)
6: (-1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8473870, 0.8473870)
7: (-1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6141288, 0.6141288)
8: (-1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1522088, 0.1522087)
9: (-0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3488287, 0.3488287)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.77 + 446.26 = 454.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0673814, upper bound: 0.0673797

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2726

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2071

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673765, upper bound: 0.0673808
time: 150.49 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673789, upper bound: 0.0673761
time: 535.53 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 686.03 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 686.03
Output dim: 9, lower bound: -0.0673765, upper bound: 0.0673808
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 686.03
Output dim: 9, lower bound: -0.0673789, upper bound: 0.0673761

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2767917, 0.2767118
1: 0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0524126, 0.0524127
2: -1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5569959, 0.5569960
3: -2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5425183, 0.5425178
4: -1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7700697, 0.7700697
5: -3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3982419, 0.3982416
6: -1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8473711, 0.8473713
7: -1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6141284, 0.6141282
8: -1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1508901, 0.1508205
9: -0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3488166, 0.3488178

Time for backsubstitution: 5.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2653

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3548

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673619, upper bound: 0.0673171
time: 336.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673170, upper bound: 0.0673645
time: 281.29 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2767118, 0.2767917
1: 0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0524127, 0.0524126
2: -1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5569960, 0.5569959
3: -2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5425178, 0.5425183
4: -1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7700697, 0.7700697
5: -3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3982416, 0.3982419
6: -1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8473713, 0.8473711
7: -1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6141282, 0.6141283
8: -1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1508205, 0.1508901
9: -0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3488178, 0.3488166

Time for backsubstitution: 5.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 510

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2582

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673714, upper bound: 0.0673699
time: 480.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673706, upper bound: 0.0673699
time: 442.18 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 928.00 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 928.00
Output dim: 9, lower bound: -0.0673619, upper bound: 0.0673171
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 928.00
Output dim: 9, lower bound: -0.0673170, upper bound: 0.0673645
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 928.00
Output dim: 9, lower bound: -0.0673714, upper bound: 0.0673699
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 928.00
Output dim: 9, lower bound: -0.0673706, upper bound: 0.0673699

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2767534, 0.2766711
1: 0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0522667, 0.0522285
2: -1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5567215, 0.5567840
3: -2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5425113, 0.5425119
4: -1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7699571, 0.7699825
5: -3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3982133, 0.3982259
6: -1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8472118, 0.8471934
7: -1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6140040, 0.6140249
8: -1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1506188, 0.1505760
9: -0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3488154, 0.3488197

Time for backsubstitution: 5.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 763

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3210

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673616, upper bound: 0.0673154
time: 744.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673609, upper bound: 0.0673130
time: 117.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2767509, 0.2766736
1: 0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0522285, 0.0522667
2: -1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5567838, 0.5567217
3: -2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5425124, 0.5425109
4: -1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7699826, 0.7699570
5: -3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3982262, 0.3982130
6: -1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8471931, 0.8472121
7: -1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6140250, 0.6140038
8: -1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1506456, 0.1505493
9: -0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3488185, 0.3488165

Time for backsubstitution: 5.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 680

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3278

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673087, upper bound: 0.0673357
time: 408.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0672853, upper bound: 0.0673566
time: 102.59 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 516.70 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 516.70
Output dim: 9, lower bound: -0.0673616, upper bound: 0.0673154
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 516.70
Output dim: 9, lower bound: -0.0673609, upper bound: 0.0673130
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 516.70
Output dim: 9, lower bound: -0.0673087, upper bound: 0.0673357
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 516.70
Output dim: 9, lower bound: -0.0672853, upper bound: 0.0673566
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 516.70
Output dim: 9, lower bound: -0.0673714, upper bound: 0.0673699
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 516.70
Output dim: 9, lower bound: -0.0673706, upper bound: 0.0673699

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 454.03 + 3621.90 = 4075.93 seconds
