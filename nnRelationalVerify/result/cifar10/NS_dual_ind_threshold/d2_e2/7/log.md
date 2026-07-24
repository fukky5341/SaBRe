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
execution time: IAR + RelationalAnalysis = 7.85 + 465.17 = 473.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0673814, upper bound: 0.0673797

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 358
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 270
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3393
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2755
type: A, layer: 1, pos: 3391
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2754
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2783
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 2781
type: A, layer: 1, pos: 3196
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2726
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 3195
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2311
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3475
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2683
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2856
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3210
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3375
type: A, layer: 1, pos: 2447
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 3222
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2954
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3464

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 358

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0671531, upper bound: 0.0673817
time: 345.66 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673804, upper bound: 0.0673822
time: 25.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 371.72 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 371.72
Output dim: 9, lower bound: -0.0671531, upper bound: 0.0673817
NS_A2, status: Status.UNKNOWN, split count: 1, time: 371.72
Output dim: 9, lower bound: -0.0673804, upper bound: 0.0673822

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1.7008055, -1.0417038, -1.7008419, -1.0416218, -0.2778693, 0.2778102
1: 0.2186840, 0.4784918, 0.2185579, 0.4784928, -0.0519885, 0.0520670
2: -1.7780809, -1.0204335, -1.7781703, -1.0198424, -0.5548396, 0.5543398
3: -2.8882585, -1.6078014, -2.8882794, -1.6075988, -0.5418860, 0.5418110
4: -1.6961937, -0.6821444, -1.6962161, -0.6819217, -0.7691813, 0.7689651
5: -3.4425411, -2.0506308, -3.4426069, -2.0500684, -0.3960473, 0.3955821
6: -1.9854085, -0.8200505, -1.9854512, -0.8197970, -0.8464713, 0.8464254
7: -1.9515333, -0.3789546, -1.9515846, -0.3785977, -0.6126236, 0.6124895
8: -1.2100987, -0.6896886, -1.2101262, -0.6896878, -0.1521396, 0.1521479
9: -0.0740916, 0.3096725, -0.0745904, 0.3096964, -0.3466241, 0.3471057

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 270
type: B, layer: 1, pos: 256
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 358
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 3548
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3393
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 3391
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2783
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2093
type: B, layer: 1, pos: 2781
type: B, layer: 1, pos: 3196
type: B, layer: 1, pos: 2726
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2208
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3195
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2311
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2856
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 3210
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 3375
type: B, layer: 1, pos: 2447
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2770
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3178
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 3222
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3177
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2954
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3364
type: B, layer: 1, pos: 3464

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 271

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0670262, upper bound: 0.0673768
time: 319.17 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0671516, upper bound: 0.0673780
time: 635.16 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.7017276, -1.0413718, -1.7009580, -1.0413722, -0.2791031, 0.2780890
1: 0.2180498, 0.4798928, 0.2181482, 0.4784955, -0.0521768, 0.0536746
2: -1.7848636, -1.0179144, -1.7784587, -1.0179302, -0.5633230, 0.5562863
3: -2.8915517, -1.6073539, -2.8883498, -1.6073990, -0.5443545, 0.5421624
4: -1.6974678, -0.6809592, -1.6962824, -0.6812015, -0.7712329, 0.7701473
5: -3.4502196, -2.0484900, -3.4428189, -2.0484250, -0.4058575, 0.3960738
6: -1.9901369, -0.8193820, -1.9855906, -0.8193073, -0.8492174, 0.8463480
7: -1.9544580, -0.3772790, -1.9517515, -0.3774405, -0.6155976, 0.6134715
8: -1.2107669, -0.6893483, -1.2101946, -0.6896860, -0.1527455, 0.1522100
9: -0.0765891, 0.3131952, -0.0762189, 0.3097728, -0.3491279, 0.3522546

Time for backsubstitution: 6.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 270
type: B, layer: 1, pos: 256
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 358
type: B, layer: 1, pos: 3548
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3393
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 3391
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2783
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2093
type: B, layer: 1, pos: 2781
type: B, layer: 1, pos: 3196
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2726
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2208
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3195
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2311
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2856
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3210
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3375
type: B, layer: 1, pos: 2447
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2770
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3178
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 3222
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3177
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2954
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3329
type: B, layer: 1, pos: 3364
type: B, layer: 1, pos: 3464

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 271

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0672528, upper bound: 0.0673778
time: 23.74 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673783, upper bound: 0.0673756
time: 452.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 482.33 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 482.33
Output dim: 9, lower bound: -0.0670262, upper bound: 0.0673768
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 482.33
Output dim: 9, lower bound: -0.0671516, upper bound: 0.0673780
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 482.33
Output dim: 9, lower bound: -0.0672528, upper bound: 0.0673778
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 482.33
Output dim: 9, lower bound: -0.0673783, upper bound: 0.0673756

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1.7005780, -1.0432975, -1.7005595, -1.0436102, -0.2755286, 0.2758460
1: 0.2190153, 0.4784847, 0.2189714, 0.4784840, -0.0516679, 0.0516725
2: -1.7779788, -1.0212708, -1.7780404, -1.0208871, -0.5520114, 0.5517724
3: -2.8882267, -1.6089990, -2.8882389, -1.6090906, -0.5404960, 0.5406717
4: -1.6940434, -0.6821916, -1.6935432, -0.6819801, -0.7673025, 0.7666683
5: -3.4424646, -2.0513604, -3.4425094, -2.0509779, -0.3940650, 0.3937836
6: -1.9851846, -0.8207939, -1.9851729, -0.8207054, -0.8450574, 0.8450876
7: -1.9512444, -0.3793619, -1.9512269, -0.3790996, -0.6117880, 0.6116548
8: -1.2095766, -0.6896890, -1.2095045, -0.6896887, -0.1514658, 0.1513523
9: -0.0740720, 0.3086712, -0.0745663, 0.3084556, -0.3452272, 0.3459702

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 270
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3393
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2755
type: A, layer: 1, pos: 3391
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2754
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2783
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 2781
type: A, layer: 1, pos: 3196
type: A, layer: 1, pos: 2726
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3195
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2311
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 3475
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2683
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2856
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3210
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3375
type: A, layer: 1, pos: 2447
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 3222
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2954
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3464

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 294

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0668434, upper bound: 0.0673761
time: 79.15 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0670245, upper bound: 0.0673743
time: 374.79 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1.7007965, -1.0417367, -1.7054870, -1.0416560, -0.2758700, 0.2825781
1: 0.2186847, 0.4784912, 0.2185510, 0.4791856, -0.0526134, 0.0517419
2: -1.7780776, -1.0207717, -1.7808422, -1.0200665, -0.5538736, 0.5547671
3: -2.8882561, -1.6078048, -2.8909190, -1.6073234, -0.5409975, 0.5441144
4: -1.6960483, -0.6821480, -1.6975007, -0.6768136, -0.7734455, 0.7695906
5: -3.4425390, -2.0506353, -3.4448066, -2.0500038, -0.3952838, 0.3964403
6: -1.9853982, -0.8201100, -1.9857050, -0.8196456, -0.8468922, 0.8462746
7: -1.9515197, -0.3790274, -1.9524534, -0.3781872, -0.6124709, 0.6129773
8: -1.2097691, -0.6896886, -1.2101518, -0.6885114, -0.1527844, 0.1520031
9: -0.0740899, 0.3096710, -0.0775638, 0.3098978, -0.3465474, 0.3503275

Time for backsubstitution: 6.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 270
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3393
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2755
type: A, layer: 1, pos: 3391
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2754
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2783
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 2781
type: A, layer: 1, pos: 3196
type: A, layer: 1, pos: 2726
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3195
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2311
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 3475
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2683
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2856
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3210
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3375
type: A, layer: 1, pos: 2447
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 3222
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2954
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3464

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 294

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669679, upper bound: 0.0673753
time: 381.81 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0671495, upper bound: 0.0673755
time: 371.96 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.7015007, -1.0429653, -1.7006756, -1.0433605, -0.2767624, 0.2761245
1: 0.2183812, 0.4798861, 0.2185618, 0.4784869, -0.0518563, 0.0532801
2: -1.7847558, -1.0187516, -1.7783226, -1.0189747, -0.5604932, 0.5537176
3: -2.8915184, -1.6085505, -2.8883080, -1.6088915, -0.5429636, 0.5410218
4: -1.6953185, -0.6810055, -1.6936097, -0.6812596, -0.7693549, 0.7678502
5: -3.4501388, -2.0492189, -3.4427166, -2.0493345, -0.4038731, 0.3942738
6: -1.9899126, -0.8201264, -1.9853125, -0.8202151, -0.8478043, 0.8450119
7: -1.9541690, -0.3776836, -1.9513946, -0.3779402, -0.6147627, 0.6126377
8: -1.2102448, -0.6893489, -1.2095731, -0.6896865, -0.1520718, 0.1514143
9: -0.0765697, 0.3121942, -0.0761947, 0.3085329, -0.3477319, 0.3511193

Time for backsubstitution: 6.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 270
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3393
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2755
type: A, layer: 1, pos: 3391
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2754
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2783
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2093
type: A, layer: 1, pos: 2781
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 3196
type: A, layer: 1, pos: 2726
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3195
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 2311
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3475
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2683
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2856
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 3210
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2447
type: A, layer: 1, pos: 3375
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2770
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3178
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 3222
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2954
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3329
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3464

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 294

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0670693, upper bound: 0.0673776
time: 180.55 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0672529, upper bound: 0.0673779
time: 364.12 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 473.01 + 3586.80 = 4059.82 seconds
