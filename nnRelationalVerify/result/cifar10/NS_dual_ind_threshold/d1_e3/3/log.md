## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.021356701


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3686162, 0.3686162)
1: (-5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4843123, 0.4843124)
2: (-0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716895, 0.1716896)
3: (-1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1034122, 0.1034122)
4: (0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0739262, 0.0739262)
5: (-1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1346687, 0.1346687)
6: (0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0268616, 0.0268616)
7: (-2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1104092, 0.1104092)
8: (-4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4115511, 0.4115511)
9: (-4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4662009, 0.4662009)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.82 + 189.28 = 197.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0213943, upper bound: 0.0213963

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 348
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2561
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3444
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 3242
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 3085
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 3217
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 3352
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 3466
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3554
type: A, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 426

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213912, upper bound: 0.0210626
time: 70.27 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213910, upper bound: 0.0213949
time: 11.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 81.53 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 81.53
Output dim: 3, lower bound: -0.0213912, upper bound: 0.0210626
NS_A2, status: Status.UNKNOWN, split count: 1, time: 81.53
Output dim: 3, lower bound: -0.0213910, upper bound: 0.0213949

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.5917544, -2.7540798, -3.5920630, -2.7533665, -0.3668050, 0.3663985
1: -5.0383964, -3.9160304, -5.0386171, -3.9154320, -0.4811549, 0.4819968
2: -0.1365270, 0.1241336, -0.1376064, 0.1241837, -0.1701487, 0.1714860
3: -1.2689371, -0.9050117, -1.2741544, -0.9043295, -0.0934331, 0.0972395
4: 0.0946629, 0.3351879, 0.0946031, 0.3353799, -0.0738640, 0.0736552
5: -1.4128594, -1.0206511, -1.4164796, -1.0201120, -0.1279598, 0.1302412
6: 0.4213780, 0.5598745, 0.4212822, 0.5605555, -0.0262354, 0.0257023
7: -2.1297286, -1.6126937, -2.1299813, -1.6105828, -0.1083473, 0.1062214
8: -4.8666215, -4.0508299, -4.8677959, -4.0505338, -0.4092756, 0.4103727
9: -4.6563711, -3.8232925, -4.6577868, -3.8230953, -0.4631230, 0.4650012

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2561
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3242
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2365

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213899, upper bound: 0.0208928
time: 23.92 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213911, upper bound: 0.0210644
time: 7.16 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.5927153, -2.7533655, -3.5927536, -2.7533658, -0.3681918, 0.3685727
1: -5.0391512, -3.9154198, -5.0392342, -3.9154198, -0.4826977, 0.4836008
2: -0.1377017, 0.1244999, -0.1377017, 0.1245006, -0.1716841, 0.1715968
3: -1.2741711, -0.8998338, -1.2741711, -0.8998306, -0.1034119, 0.1023654
4: 0.0945320, 0.3353948, 0.0945311, 0.3353949, -0.0737724, 0.0738877
5: -1.4164877, -1.0170825, -1.4164877, -1.0170807, -0.1346636, 0.1340413
6: 0.4208487, 0.5605581, 0.4208483, 0.5605581, -0.0267781, 0.0268609
7: -2.1313541, -1.6105542, -2.1313646, -1.6105542, -0.1100072, 0.1104084
8: -4.8678236, -4.0498376, -4.8678236, -4.0498276, -0.4115507, 0.4109638
9: -4.6579099, -3.8221581, -4.6579094, -3.8221426, -0.4661952, 0.4642770

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2561
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3242
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2365

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213903, upper bound: 0.0212277
time: 8.11 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213896, upper bound: 0.0213944
time: 16.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.21 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 30.21
Output dim: 3, lower bound: -0.0213899, upper bound: 0.0208928
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.21
Output dim: 3, lower bound: -0.0213911, upper bound: 0.0210644
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.21
Output dim: 3, lower bound: -0.0213903, upper bound: 0.0212277
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.21
Output dim: 3, lower bound: -0.0213896, upper bound: 0.0213944

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.5917501, -2.7618511, -3.5920584, -2.7620800, -0.3566581, 0.3574471
1: -5.0383959, -3.9255517, -5.0386167, -3.9260020, -0.4706474, 0.4725673
2: -0.1364857, 0.1240067, -0.1375588, 0.1240386, -0.1699146, 0.1712684
3: -1.2668651, -0.9050119, -1.2718155, -0.9043297, -0.0913591, 0.0950180
4: 0.0947049, 0.3351054, 0.0946510, 0.3352883, -0.0736270, 0.0734255
5: -1.4104167, -1.0206516, -1.4137684, -1.0201128, -0.1255576, 0.1276135
6: 0.4213821, 0.5598475, 0.4212869, 0.5605249, -0.0261923, 0.0256622
7: -2.1296489, -1.6126959, -2.1298902, -1.6105855, -0.1081612, 0.1060141
8: -4.8666201, -4.0573616, -4.8677945, -4.0578170, -0.4013978, 0.4032905
9: -4.6563373, -3.8278077, -4.6577482, -3.8282371, -0.4577002, 0.4601969

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 348
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2561
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3444
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 3242
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 3085
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 3217
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 3352
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 3466
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3554
type: A, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2572

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213582, upper bound: 0.0207258
time: 10.29 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213590, upper bound: 0.0208654
time: 5.63 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.5917540, -2.7541265, -3.6009636, -2.7534204, -0.3572505, 0.3768101
1: -5.0383959, -3.9160800, -5.0494919, -3.9154894, -0.4713533, 0.4928708
2: -0.1364798, 0.1240830, -0.1377267, 0.1241258, -0.1699236, 0.1715742
3: -1.2689071, -0.9050117, -1.2741197, -0.9019374, -0.0957740, 0.0951646
4: 0.0946709, 0.3351194, 0.0946087, 0.3353061, -0.0736712, 0.0737657
5: -1.4128269, -1.0206511, -1.4164420, -1.0172868, -0.1307562, 0.1277925
6: 0.4213824, 0.5598588, 0.4212744, 0.5605378, -0.0261908, 0.0257143
7: -2.1295984, -1.6126951, -2.1298332, -1.6106379, -0.1084873, 0.1060084
8: -4.8666215, -4.0508699, -4.8752766, -4.0505786, -0.4018886, 0.4185716
9: -4.6563325, -3.8233867, -4.6632671, -3.8232031, -0.4576663, 0.4698366

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 348
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2561
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3444
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 3242
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 3085
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 3217
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 3352
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 3466
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3554
type: A, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2572

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213596, upper bound: 0.0208927
time: 63.33 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213591, upper bound: 0.0210291
time: 154.28 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.5927117, -2.7611370, -3.5927496, -2.7620792, -0.3580449, 0.3596214
1: -5.0391512, -3.9249413, -5.0392337, -3.9259899, -0.4721895, 0.4741722
2: -0.1376599, 0.1243730, -0.1376539, 0.1243555, -0.1714492, 0.1713785
3: -1.2720997, -0.8998337, -1.2718316, -0.8998308, -0.1013400, 0.1001433
4: 0.0945739, 0.3353122, 0.0945789, 0.3353032, -0.0735354, 0.0736580
5: -1.4140460, -1.0170829, -1.4137764, -1.0170814, -0.1322622, 0.1314131
6: 0.4208528, 0.5605313, 0.4208529, 0.5605274, -0.0267351, 0.0268208
7: -2.1312749, -1.6105564, -2.1312737, -1.6105567, -0.1098210, 0.1102012
8: -4.8678222, -4.0563698, -4.8678226, -4.0571108, -0.4036729, 0.4038813
9: -4.6578760, -3.8266728, -4.6578717, -3.8272851, -0.4607727, 0.4594731

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 348
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2561
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3444
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 3242
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 3085
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 3217
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 3352
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 3466
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3554
type: A, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2572

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213589, upper bound: 0.0210558
time: 6.66 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213600, upper bound: 0.0211917
time: 48.11 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.5927150, -2.7534127, -3.6016543, -2.7534194, -0.3586374, 0.3789845
1: -5.0391502, -3.9154706, -5.0501080, -3.9154778, -0.4728955, 0.4944751
2: -0.1376542, 0.1244493, -0.1378222, 0.1244427, -0.1714581, 0.1716847
3: -1.2741408, -0.8998338, -1.2741362, -0.8974383, -0.1057528, 0.1002899
4: 0.0945401, 0.3353264, 0.0945365, 0.3353211, -0.0735795, 0.0739983
5: -1.4164548, -1.0170823, -1.4164501, -1.0142550, -0.1374599, 0.1315921
6: 0.4208531, 0.5605426, 0.4208405, 0.5605403, -0.0267336, 0.0268725
7: -2.1312244, -1.6105560, -2.1312172, -1.6106094, -0.1101473, 0.1101954
8: -4.8678226, -4.0498776, -4.8753047, -4.0498729, -0.4041638, 0.4191623
9: -4.6578717, -3.8222537, -4.6633897, -3.8222501, -0.4607393, 0.4691119

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 348
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2561
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 3444
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 3242
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 3085
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 3217
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 3352
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 3466
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 2932
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3554
type: A, layer: 1, pos: 3591

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2572

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213596, upper bound: 0.0210548
time: 107.50 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213589, upper bound: 0.0213585
time: 29.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 143.36 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 143.36
Output dim: 3, lower bound: -0.0213582, upper bound: 0.0207258
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 143.36
Output dim: 3, lower bound: -0.0213590, upper bound: 0.0208654
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 143.36
Output dim: 3, lower bound: -0.0213596, upper bound: 0.0208927
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 143.36
Output dim: 3, lower bound: -0.0213591, upper bound: 0.0210291
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 143.36
Output dim: 3, lower bound: -0.0213589, upper bound: 0.0210558
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 143.36
Output dim: 3, lower bound: -0.0213600, upper bound: 0.0211917
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 143.36
Output dim: 3, lower bound: -0.0213596, upper bound: 0.0210548
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 143.36
Output dim: 3, lower bound: -0.0213589, upper bound: 0.0213585

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.5877128, -2.7618515, -3.5886374, -2.7620802, -0.3520547, 0.3534970
1: -5.0324602, -3.9255550, -5.0335951, -3.9260056, -0.4631030, 0.4661385
2: -0.1363556, 0.1240067, -0.1374480, 0.1240386, -0.1697429, 0.1711230
3: -1.2668586, -0.9062158, -1.2718096, -0.9053487, -0.0900913, 0.0935323
4: 0.0947124, 0.3351018, 0.0946575, 0.3352854, -0.0736090, 0.0734073
5: -1.4104137, -1.0221108, -1.4137659, -1.0213466, -0.1240060, 0.1258156
6: 0.4214109, 0.5598475, 0.4213111, 0.5605248, -0.0261666, 0.0256390
7: -2.1296489, -1.6129758, -2.1298902, -1.6108222, -0.1078936, 0.1057036
8: -4.8653193, -4.0573735, -4.8666954, -4.0578270, -0.3997321, 0.4018356
9: -4.6548686, -3.8278074, -4.6565008, -3.8282368, -0.4560337, 0.4586681

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2561
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3242
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2393

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213546, upper bound: 0.0205710
time: 41.13 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213530, upper bound: 0.0207168
time: 283.58 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.5905175, -2.7523980, -3.5909905, -2.7620802, -0.3524756, 0.3685802
1: -5.0356283, -3.9114184, -5.0361910, -3.9260049, -0.4636834, 0.4909953
2: -0.1364840, 0.1240670, -0.1373668, 0.1240386, -0.1699596, 0.1713478
3: -1.2697525, -0.9056426, -1.2718111, -0.9048690, -0.0950781, 0.0936438
4: 0.0939224, 0.3351029, 0.0946579, 0.3352811, -0.0745018, 0.0734038
5: -1.4138738, -1.0214689, -1.4137666, -1.0208136, -0.1300737, 0.1259475
6: 0.4213796, 0.5599158, 0.4213049, 0.5605248, -0.0262147, 0.0257217
7: -2.1302524, -1.6128583, -2.1298902, -1.6107299, -0.1089257, 0.1057332
8: -4.8658690, -4.0544510, -4.8671436, -4.0578289, -0.3998839, 0.4068641
9: -4.6556587, -3.8245409, -4.6571140, -3.8282368, -0.4564494, 0.4633820

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2561
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3242
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2393

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213550, upper bound: 0.0207129
time: 7.72 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213554, upper bound: 0.0208589
time: 20.51 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.5877161, -2.7541268, -3.5975420, -2.7534208, -0.3526471, 0.3728600
1: -5.0324602, -3.9160838, -5.0444703, -3.9154940, -0.4638089, 0.4864422
2: -0.1363498, 0.1240830, -0.1376161, 0.1241258, -0.1697519, 0.1714287
3: -1.2689005, -0.9062157, -1.2741143, -0.9029564, -0.0945060, 0.0936788
4: 0.0946788, 0.3351161, 0.0946152, 0.3353033, -0.0736532, 0.0737474
5: -1.4128239, -1.0221108, -1.4164397, -1.0185205, -0.1292044, 0.1259945
6: 0.4214112, 0.5598588, 0.4212987, 0.5605378, -0.0261651, 0.0256911
7: -2.1295984, -1.6129755, -2.1298332, -1.6108750, -0.1082198, 0.1056979
8: -4.8653202, -4.0508819, -4.8741779, -4.0505896, -0.4002228, 0.4171168
9: -4.6548634, -3.8233876, -4.6620193, -3.8232026, -0.4559995, 0.4683075

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2561
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3242
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2393

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213562, upper bound: 0.0207386
time: 47.05 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213540, upper bound: 0.0208877
time: 9.02 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.5905211, -2.7446728, -3.5998955, -2.7534206, -0.3530679, 0.3879431
1: -5.0356274, -3.9019480, -5.0470657, -3.9154925, -0.4643893, 0.5112981
2: -0.1364782, 0.1241433, -0.1375348, 0.1241258, -0.1699684, 0.1716536
3: -1.2717938, -0.9056427, -1.2741157, -0.9024765, -0.0994909, 0.0937904
4: 0.0938888, 0.3351172, 0.0946158, 0.3352989, -0.0745461, 0.0737439
5: -1.4162824, -1.0214686, -1.4164405, -1.0179874, -0.1352714, 0.1261265
6: 0.4213799, 0.5599270, 0.4212925, 0.5605377, -0.0262132, 0.0257737
7: -2.1302018, -1.6128579, -2.1298332, -1.6107829, -0.1092518, 0.1057275
8: -4.8658695, -4.0479574, -4.8746266, -4.0505905, -0.4003746, 0.4221454
9: -4.6556535, -3.8201210, -4.6626334, -3.8232031, -0.4564152, 0.4730213

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2561
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3242
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2393

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213546, upper bound: 0.0208812
time: 10.47 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213553, upper bound: 0.0210288
time: 7.18 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.5886738, -2.7611377, -3.5893281, -2.7620797, -0.3534412, 0.3556713
1: -5.0332160, -3.9249454, -5.0342126, -3.9259932, -0.4646454, 0.4677434
2: -0.1375301, 0.1243730, -0.1375432, 0.1243555, -0.1712781, 0.1712333
3: -1.2720933, -0.9010378, -1.2718265, -0.9008497, -0.1000721, 0.0986576
4: 0.0945817, 0.3353089, 0.0945854, 0.3353003, -0.0735175, 0.0736398
5: -1.4140429, -1.0185423, -1.4137739, -1.0183151, -0.1307103, 0.1296153
6: 0.4208814, 0.5605312, 0.4208772, 0.5605274, -0.0267095, 0.0267976
7: -2.1312749, -1.6108367, -2.1312737, -1.6107931, -0.1095534, 0.1098917
8: -4.8665223, -4.0563817, -4.8667221, -4.0571213, -0.4020085, 0.4024263
9: -4.6564069, -3.8266728, -4.6566248, -3.8272848, -0.4591052, 0.4579444

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2561
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3242
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2393

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213550, upper bound: 0.0209038
time: 42.61 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213545, upper bound: 0.0210474
time: 82.84 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.5914788, -2.7516840, -3.5916815, -2.7620792, -0.3538620, 0.3707543
1: -5.0363832, -3.9108086, -5.0368080, -3.9259923, -0.4652255, 0.4926000
2: -0.1376608, 0.1244333, -0.1374621, 0.1243555, -0.1714979, 0.1714581
3: -1.2749883, -0.9004650, -1.2718277, -0.9003702, -0.1050577, 0.0987691
4: 0.0937923, 0.3353103, 0.0945857, 0.3352961, -0.0744100, 0.0736365
5: -1.4175037, -1.0179005, -1.4137745, -1.0177817, -0.1367775, 0.1297472
6: 0.4208504, 0.5605994, 0.4208710, 0.5605273, -0.0267573, 0.0268792
7: -2.1318781, -1.6107185, -2.1312737, -1.6107012, -0.1105857, 0.1099214
8: -4.8670712, -4.0534582, -4.8671718, -4.0571227, -0.4021589, 0.4074553
9: -4.6571970, -3.8234074, -4.6572380, -3.8272851, -0.4595205, 0.4626575

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2561
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3242
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2393

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213550, upper bound: 0.0210413
time: 10.91 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213546, upper bound: 0.0211905
time: 6.56 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.5886772, -2.7534127, -3.5982330, -2.7534199, -0.3540335, 0.3750343
1: -5.0332160, -3.9154744, -5.0450873, -3.9154816, -0.4653512, 0.4880461
2: -0.1375244, 0.1244493, -0.1377115, 0.1244427, -0.1712869, 0.1715392
3: -1.2741345, -0.9010378, -1.2741309, -0.8984573, -0.1044848, 0.0988041
4: 0.0945479, 0.3353231, 0.0945431, 0.3353183, -0.0735616, 0.0739799
5: -1.4164518, -1.0185418, -1.4164476, -1.0154886, -0.1359081, 0.1297943
6: 0.4208817, 0.5605425, 0.4208648, 0.5605403, -0.0267080, 0.0268492
7: -2.1312244, -1.6108364, -2.1312172, -1.6108463, -0.1098796, 0.1098859
8: -4.8665228, -4.0498896, -4.8742056, -4.0498834, -0.4024993, 0.4177075
9: -4.6564021, -3.8222537, -4.6621418, -3.8222504, -0.4590713, 0.4675826

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2561
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3242
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2393

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213556, upper bound: 0.0210680
time: 114.24 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213554, upper bound: 0.0212192
time: 49.19 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.5914826, -2.7439590, -3.6005862, -2.7534194, -0.3544543, 0.3901173
1: -5.0363827, -3.9013379, -5.0476823, -3.9154797, -0.4659314, 0.5129016
2: -0.1376552, 0.1245096, -0.1376302, 0.1244427, -0.1715065, 0.1717640
3: -1.2770288, -0.9004649, -1.2741321, -0.8979775, -0.1072687, 0.0989157
4: 0.0937583, 0.3353245, 0.0945435, 0.3353138, -0.0744544, 0.0739767
5: -1.4199109, -1.0179002, -1.4164484, -1.0149555, -0.1419745, 0.1299262
6: 0.4208508, 0.5606108, 0.4208586, 0.5605403, -0.0267558, 0.0269309
7: -2.1318278, -1.6107183, -2.1312172, -1.6107539, -0.1109119, 0.1099156
8: -4.8670712, -4.0469656, -4.8746548, -4.0498843, -0.4026499, 0.4227365
9: -4.6571918, -3.8189874, -4.6627560, -3.8222508, -0.4594867, 0.4722958

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2561
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3242
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2932
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2393

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213524, upper bound: 0.0212108
time: 7.65 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213558, upper bound: 0.0213560
time: 128.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 142.34 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213546, upper bound: 0.0205710
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213530, upper bound: 0.0207168
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213550, upper bound: 0.0207129
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213554, upper bound: 0.0208589
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213562, upper bound: 0.0207386
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213540, upper bound: 0.0208877
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213546, upper bound: 0.0208812
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213553, upper bound: 0.0210288
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213550, upper bound: 0.0209038
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213545, upper bound: 0.0210474
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213550, upper bound: 0.0210413
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213546, upper bound: 0.0211905
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213556, upper bound: 0.0210680
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213554, upper bound: 0.0212192
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213524, upper bound: 0.0212108
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 142.34
Output dim: 3, lower bound: -0.0213558, upper bound: 0.0213560

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 197.10 + 1516.25 = 1713.35 seconds
