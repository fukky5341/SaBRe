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
execution time: IAR + RelationalAnalysis = 8.44 + 456.26 = 464.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0673814, upper bound: 0.0673797

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3548

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 537

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673810, upper bound: 0.0673703
time: 413.69 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673696, upper bound: 0.0673826
time: 59.16 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 472.92 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 472.92
Output dim: 9, lower bound: -0.0673810, upper bound: 0.0673703
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 472.92
Output dim: 9, lower bound: -0.0673696, upper bound: 0.0673826

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2783218, 0.2783219
1: 0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0524148, 0.0524148
2: -1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5569974, 0.5569974
3: -2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5425168, 0.5425162
4: -1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7700785, 0.7700785
5: -3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3982364, 0.3982357
6: -1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8473871, 0.8473870
7: -1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6141288, 0.6141287
8: -1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1522081, 0.1522082
9: -0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3488287, 0.3488287

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3548

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 396

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673179, upper bound: 0.0673661
time: 176.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673773, upper bound: 0.0673070
time: 578.43 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2783220, 0.2783218
1: 0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0524148, 0.0524148
2: -1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5569974, 0.5569974
3: -2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5425162, 0.5425168
4: -1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7700784, 0.7700785
5: -3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3982357, 0.3982365
6: -1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8473871, 0.8473870
7: -1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6141287, 0.6141288
8: -1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1522082, 0.1522081
9: -0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3488287, 0.3488287

Time for backsubstitution: 6.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3548

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 396

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673061, upper bound: 0.0673777
time: 183.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673663, upper bound: 0.0673173
time: 294.26 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 484.10 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 484.10
Output dim: 9, lower bound: -0.0673179, upper bound: 0.0673661
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 484.10
Output dim: 9, lower bound: -0.0673773, upper bound: 0.0673070
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 484.10
Output dim: 9, lower bound: -0.0673061, upper bound: 0.0673777
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 484.10
Output dim: 9, lower bound: -0.0673663, upper bound: 0.0673173

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2777850, 0.2777362
1: 0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0507234, 0.0509311
2: -1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5564759, 0.5563515
3: -2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5406336, 0.5402814
4: -1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7699025, 0.7699009
5: -3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3942865, 0.3937866
6: -1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8437137, 0.8441676
7: -1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6084681, 0.6076236
8: -1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1508981, 0.1510587
9: -0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3485217, 0.3485597

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3548

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3278

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673112, upper bound: 0.0673361
time: 28.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0672873, upper bound: 0.0673604
time: 47.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2777361, 0.2777852
1: 0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0509311, 0.0507234
2: -1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5563515, 0.5564759
3: -2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5402821, 0.5406330
4: -1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7699009, 0.7699025
5: -3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3937874, 0.3942858
6: -1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8441676, 0.8437137
7: -1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6076238, 0.6084680
8: -1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1510587, 0.1508981
9: -0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3485597, 0.3485217

Time for backsubstitution: 6.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3548

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3278

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673727, upper bound: 0.0672756
time: 39.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673474, upper bound: 0.0673007
time: 357.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2777852, 0.2777361
1: 0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0507234, 0.0509311
2: -1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5564759, 0.5563515
3: -2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5406330, 0.5402821
4: -1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7699025, 0.7699009
5: -3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3942858, 0.3937874
6: -1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8437138, 0.8441676
7: -1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6084680, 0.6076238
8: -1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1508981, 0.1510587
9: -0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3485217, 0.3485597

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3548

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3278

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0672999, upper bound: 0.0673484
time: 31.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0672751, upper bound: 0.0673717
time: 262.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2777362, 0.2777850
1: 0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0509311, 0.0507234
2: -1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5563515, 0.5564759
3: -2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5402814, 0.5406336
4: -1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7699009, 0.7699025
5: -3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3937866, 0.3942866
6: -1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8441676, 0.8437137
7: -1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6076237, 0.6084681
8: -1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1510587, 0.1508981
9: -0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3485597, 0.3485217

Time for backsubstitution: 6.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3548

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3278

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673591, upper bound: 0.0672853
time: 35.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673357, upper bound: 0.0673020
time: 242.71 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 285.28 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 285.28
Output dim: 9, lower bound: -0.0673112, upper bound: 0.0673361
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 285.28
Output dim: 9, lower bound: -0.0672873, upper bound: 0.0673604
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 285.28
Output dim: 9, lower bound: -0.0673727, upper bound: 0.0672756
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 285.28
Output dim: 9, lower bound: -0.0673474, upper bound: 0.0673007
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 285.28
Output dim: 9, lower bound: -0.0672999, upper bound: 0.0673484
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 285.28
Output dim: 9, lower bound: -0.0672751, upper bound: 0.0673717
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 285.28
Output dim: 9, lower bound: -0.0673591, upper bound: 0.0672853
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 285.28
Output dim: 9, lower bound: -0.0673357, upper bound: 0.0673020

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.7009605, -1.0413542, -1.7009605, -1.0413542, -0.2773195, 0.2772530
1: 0.2181461, 0.4784963, 0.2181461, 0.4784963, -0.0506375, 0.0508400
2: -1.7784593, -1.0179119, -1.7784593, -1.0179119, -0.5561945, 0.5560744
3: -2.8883500, -1.6066654, -2.8883500, -1.6066654, -0.5387442, 0.5384340
4: -1.6963110, -0.6811960, -1.6963110, -0.6811960, -0.7701007, 0.7701086
5: -3.4428194, -2.0479853, -3.4428194, -2.0479853, -0.3921884, 0.3917260
6: -1.9855907, -0.8188132, -1.9855907, -0.8188132, -0.8435891, 0.8440453
7: -1.9517522, -0.3774295, -1.9517522, -0.3774295, -0.6080688, 0.6072861
8: -1.2102182, -0.6896860, -1.2102182, -0.6896860, -0.1506088, 0.1507431
9: -0.0762199, 0.3097731, -0.0762199, 0.3097731, -0.3485269, 0.3485645

Time for backsubstitution: 6.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 358
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2770
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2856
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3222
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 270
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2726
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2783
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 3178
type: DSZ, layer: 1, pos: 3195
type: DSZ, layer: 1, pos: 3196
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3210
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3329
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3391
type: DSZ, layer: 1, pos: 3393
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 3548

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3326

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673058, upper bound: 0.0673353
time: 34.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673090, upper bound: 0.0673313
time: 484.73 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 525.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 525.58
Output dim: 9, lower bound: -0.0673058, upper bound: 0.0673353
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 525.58
Output dim: 9, lower bound: -0.0673090, upper bound: 0.0673313
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 525.58
Output dim: 9, lower bound: -0.0672873, upper bound: 0.0673604
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 525.58
Output dim: 9, lower bound: -0.0673727, upper bound: 0.0672756
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 525.58
Output dim: 9, lower bound: -0.0673474, upper bound: 0.0673007
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 525.58
Output dim: 9, lower bound: -0.0672999, upper bound: 0.0673484
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 525.58
Output dim: 9, lower bound: -0.0672751, upper bound: 0.0673717
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 525.58
Output dim: 9, lower bound: -0.0673591, upper bound: 0.0672853
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 525.58
Output dim: 9, lower bound: -0.0673357, upper bound: 0.0673020

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 464.71 + 3315.36 = 3780.07 seconds
