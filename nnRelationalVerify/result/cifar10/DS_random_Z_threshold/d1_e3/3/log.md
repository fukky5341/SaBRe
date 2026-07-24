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
execution time: IAR + RelationalAnalysis = 7.93 + 186.54 = 194.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0213943, upper bound: 0.0213963

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2465

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2202

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213922, upper bound: 0.0213925
time: 54.86 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213891, upper bound: 0.0213967
time: 5.34 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 60.21 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 60.21
Output dim: 3, lower bound: -0.0213922, upper bound: 0.0213925
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 60.21
Output dim: 3, lower bound: -0.0213891, upper bound: 0.0213967

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3661618, 0.3661285
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4786168, 0.4784876
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716885, 0.1716891
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1030997, 0.1030941
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0738053, 0.0738075
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1345266, 0.1345287
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267876, 0.0267871
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1092119, 0.1092418
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4049408, 0.4047775
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4601861, 0.4600517

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2262

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3085

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213657, upper bound: 0.0213805
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213781, upper bound: 0.0213629
time: 144.86 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3661285, 0.3661618
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4784876, 0.4786168
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716891, 0.1716885
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1030941, 0.1030997
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0738075, 0.0738053
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1345287, 0.1345266
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267871, 0.0267876
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1092418, 0.1092119
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4047776, 0.4049407
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4600516, 0.4601861

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3385

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 737

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213840, upper bound: 0.0213942
time: 8.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213859, upper bound: 0.0213903
time: 30.21 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 45.02 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 45.02
Output dim: 3, lower bound: -0.0213657, upper bound: 0.0213805
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 45.02
Output dim: 3, lower bound: -0.0213781, upper bound: 0.0213629
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 45.02
Output dim: 3, lower bound: -0.0213840, upper bound: 0.0213942
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 45.02
Output dim: 3, lower bound: -0.0213859, upper bound: 0.0213903

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3555040, 0.3552635
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4621483, 0.4614601
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1713487, 0.1713362
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1019812, 0.1019884
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0737148, 0.0737198
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1326855, 0.1327162
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267163, 0.0267136
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1081636, 0.1082287
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.3896146, 0.3889176
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4475528, 0.4469751

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 674

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 865

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213640, upper bound: 0.0213780
time: 4.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213629, upper bound: 0.0213734
time: 41.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3552969, 0.3554707
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4615894, 0.4620191
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1713356, 0.1713493
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1019941, 0.1019756
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0737176, 0.0737171
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1327141, 0.1326876
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267141, 0.0267159
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1081989, 0.1081935
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.3890807, 0.3894515
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4471096, 0.4474183

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2043

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213802, upper bound: 0.0213582
time: 26.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213735, upper bound: 0.0213650
time: 55.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3659316, 0.3659469
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4782408, 0.4783460
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716865, 0.1716858
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1030370, 0.1030449
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0738070, 0.0738048
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1344826, 0.1344833
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267868, 0.0267872
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1092416, 0.1092117
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4047111, 0.4048619
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4599329, 0.4600519

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2691

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 669

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213842, upper bound: 0.0213924
time: 58.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213842, upper bound: 0.0213921
time: 13.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3659136, 0.3659650
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4782168, 0.4783700
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716864, 0.1716858
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1030393, 0.1030427
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0738070, 0.0738048
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1344854, 0.1344806
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267868, 0.0267872
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1092416, 0.1092117
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4046988, 0.4048743
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4599174, 0.4600675

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3385

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 613

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213856, upper bound: 0.0213159
time: 6.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213123, upper bound: 0.0213874
time: 21.76 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 34.47 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.47
Output dim: 3, lower bound: -0.0213640, upper bound: 0.0213780
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.47
Output dim: 3, lower bound: -0.0213629, upper bound: 0.0213734
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.47
Output dim: 3, lower bound: -0.0213802, upper bound: 0.0213582
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.47
Output dim: 3, lower bound: -0.0213735, upper bound: 0.0213650
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.47
Output dim: 3, lower bound: -0.0213842, upper bound: 0.0213924
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.47
Output dim: 3, lower bound: -0.0213842, upper bound: 0.0213921
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.47
Output dim: 3, lower bound: -0.0213856, upper bound: 0.0213159
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.47
Output dim: 3, lower bound: -0.0213123, upper bound: 0.0213874

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3543613, 0.3540877
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4581021, 0.4573501
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1713232, 0.1713106
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1016236, 0.1016301
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0735996, 0.0736050
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1326395, 0.1326710
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0266859, 0.0266833
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1072912, 0.1073573
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.3852246, 0.3844167
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4432479, 0.4425844

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 447

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213633, upper bound: 0.0213781
time: 4.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213633, upper bound: 0.0213766
time: 8.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3543282, 0.3541208
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4580384, 0.4574139
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1713231, 0.1713107
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1016229, 0.1016309
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0735999, 0.0736046
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1326403, 0.1326702
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0266860, 0.0266833
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1072921, 0.1073563
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.3851137, 0.3845275
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4431622, 0.4426701

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3044

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2572

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213311, upper bound: 0.0212069
time: 11.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0211927, upper bound: 0.0213441
time: 58.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3552886, 0.3554601
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4615070, 0.4619242
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1712735, 0.1712953
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1019739, 0.1019519
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0736945, 0.0736977
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1327147, 0.1326869
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0266859, 0.0266913
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1080900, 0.1080680
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.3889027, 0.3892472
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4469884, 0.4472641

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 527

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2335

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213281, upper bound: 0.0213316
time: 168.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213548, upper bound: 0.0213071
time: 32.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3552864, 0.3554624
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4614944, 0.4619367
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1712815, 0.1712873
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1019704, 0.1019554
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0736983, 0.0736939
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1327134, 0.1326882
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0266895, 0.0266877
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1080734, 0.1080846
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.3888764, 0.3892734
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4469554, 0.4472972

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3385

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3044

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213753, upper bound: 0.0213625
time: 66.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213743, upper bound: 0.0213659
time: 7.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3659316, 0.3659469
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4782408, 0.4783460
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716865, 0.1716858
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1030370, 0.1030449
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0738070, 0.0738048
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1344826, 0.1344833
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267868, 0.0267872
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1092416, 0.1092117
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4047111, 0.4048619
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4599329, 0.4600519

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2551

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2642

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213785, upper bound: 0.0213786
time: 88.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213700, upper bound: 0.0213764
time: 241.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3659316, 0.3659469
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4782408, 0.4783460
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716865, 0.1716858
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1030370, 0.1030449
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0738070, 0.0738048
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1344826, 0.1344833
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267868, 0.0267872
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1092416, 0.1092117
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4047111, 0.4048619
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4599329, 0.4600519

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 203

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213828, upper bound: 0.0213683
time: 65.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213616, upper bound: 0.0213921
time: 11.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3656207, 0.3657385
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4780949, 0.4782193
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716783, 0.1716786
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1026379, 0.1025862
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0737272, 0.0737344
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1342478, 0.1342125
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267046, 0.0266941
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1089335, 0.1089426
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4046953, 0.4048567
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4599192, 0.4600668

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 564

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 461

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213846, upper bound: 0.0213063
time: 7.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213759, upper bound: 0.0213153
time: 9.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3656874, 0.3656721
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4780661, 0.4782482
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716792, 0.1716777
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1025828, 0.1026413
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0737367, 0.0737250
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1342173, 0.1342430
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0266937, 0.0267051
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1089724, 0.1089036
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4046812, 0.4048709
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4599167, 0.4600692

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 706

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213120, upper bound: 0.0213146
time: 124.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213115, upper bound: 0.0213901
time: 15.50 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 146.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213633, upper bound: 0.0213781
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213633, upper bound: 0.0213766
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213311, upper bound: 0.0212069
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0211927, upper bound: 0.0213441
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213281, upper bound: 0.0213316
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213548, upper bound: 0.0213071
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213753, upper bound: 0.0213625
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213743, upper bound: 0.0213659
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213785, upper bound: 0.0213786
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213700, upper bound: 0.0213764
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213828, upper bound: 0.0213683
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213616, upper bound: 0.0213921
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213846, upper bound: 0.0213063
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213759, upper bound: 0.0213153
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213120, upper bound: 0.0213146
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 146.40
Output dim: 3, lower bound: -0.0213115, upper bound: 0.0213901

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3543613, 0.3540877
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4581021, 0.4573501
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1713232, 0.1713106
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1016236, 0.1016301
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0735996, 0.0736050
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1326395, 0.1326710
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0266859, 0.0266833
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1072912, 0.1073573
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.3852246, 0.3844167
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4432479, 0.4425844

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3023

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213468, upper bound: 0.0213582
time: 60.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213517, upper bound: 0.0213583
time: 9.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3543613, 0.3540877
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4581021, 0.4573501
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1713232, 0.1713106
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1016236, 0.1016301
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0735996, 0.0736050
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1326395, 0.1326710
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0266859, 0.0266833
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1072912, 0.1073573
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.3852246, 0.3844167
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4432479, 0.4425844

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3483

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3444

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213627, upper bound: 0.0213372
time: 8.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0213240, upper bound: 0.0213352
time: 83.37 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 98.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 98.07
Output dim: 3, lower bound: -0.0213468, upper bound: 0.0213582
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 98.07
Output dim: 3, lower bound: -0.0213517, upper bound: 0.0213583
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 98.07
Output dim: 3, lower bound: -0.0213627, upper bound: 0.0213372
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 98.07
Output dim: 3, lower bound: -0.0213240, upper bound: 0.0213352
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 98.07
Output dim: 3, lower bound: -0.0213753, upper bound: 0.0213625
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 98.07
Output dim: 3, lower bound: -0.0213743, upper bound: 0.0213659
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 98.07
Output dim: 3, lower bound: -0.0213785, upper bound: 0.0213786
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 98.07
Output dim: 3, lower bound: -0.0213700, upper bound: 0.0213764
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 98.07
Output dim: 3, lower bound: -0.0213828, upper bound: 0.0213683
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 98.07
Output dim: 3, lower bound: -0.0213616, upper bound: 0.0213921
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 98.07
Output dim: 3, lower bound: -0.0213846, upper bound: 0.0213063
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 98.07
Output dim: 3, lower bound: -0.0213759, upper bound: 0.0213153
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 98.07
Output dim: 3, lower bound: -0.0213115, upper bound: 0.0213901

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 194.46 + 1664.22 = 1858.68 seconds
